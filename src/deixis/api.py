"""The main entry: `resolve(request, profile) -> Response` (plan 4.5 to 4.8), and `adjust`.

A request is everything for one utterance: the words with their times, the pointer samples
(or holds made elsewhere: clicks, the robot's own tracker), the things in the scene and the
agent's tool calls, with references instead of coordinates. The response gives, for every
reference argument, a `Where` (place, object) or a `Size`, or one `Ask`.

What `resolve` does:

1. Validate each call against the profile (`parse_call`; a bad call raises `CallError`).
2. Holds and spans: the request's own, plus those tracked from its samples.
3. Holds that are mostly a pinch are not used as places.
4. All pointing words of all calls are matched to holds together, in time order, so "a
   house here and a tree there" gets two different holds.
5. Per call: objects first, then sizes, then places (a place may need the size), each shaped
   to what the robot wants (a point, a box, a path).
6. A place outside the work area is kept and flagged, or asked about; never moved.
7. The first Ask stops the request; what was already resolved comes back in `partial`.
8. Answers to an earlier Ask short-cut their argument.

Pure: no clocks, no randomness, no state between calls.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from deixis.actions import Call, Param, parse_call
from deixis.profile import Profile
from deixis.refs import (
    DeicticObject,
    DeicticPlace,
    DeicticSize,
    EmptySpace,
    LabelObject,
    LastObject,
    NamedPlace,
    PointedPlace,
    PresetSize,
    RelationPlace,
    RelativeSize,
    TwoPlaces,
)
from deixis.resolver import (
    Ask as TimingAsk,
)
from deixis.resolver import (
    Pick,
    ResolverParams,
    SpanHold,
    assign_holds,
    hold_where,
    last_hold,
    pick_span,
    size_holds,
)
from deixis.scene import (
    Thing,
    Unsupported,
    empty_space,
    find,
    inside_size,
    labels,
    last,
    named,
    object_at,
    relation,
)
from deixis.tracker import track
from deixis.types import Hold, Sample, Word
from deixis.where import Answer, Ask, Size, Where, Wire

__all__ = [
    "Context",
    "RawCall",
    "Request",
    "RequestError",
    "ResolvedCall",
    "Response",
    "adjust",
    "adjust_size",
    "resolve",
]


class RequestError(ValueError):
    """The request itself is malformed (a robot-side mistake, not the agent's)."""


class RawCall(BaseModel):
    """One tool call as the agent wrote it: a name and raw JSON arguments."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    args: dict[str, Any] | str | None = Field(default_factory=dict)


class Context(BaseModel):
    model_config = ConfigDict(extra="ignore")

    last_thing: str | None = Field(
        default=None, description="The id 'it' means, if the robot knows better than 'newest'."
    )
    previous: dict[str, Where | Size] = Field(
        default_factory=dict,
        description='What is being previewed, for adjust: {"place": Where, "size": Size}.',
    )


class Request(Wire):
    """Everything for one utterance (plan 4.5)."""

    words: list[Word] = Field(default_factory=list, description="This utterance, in order.")
    samples: list[Sample] = Field(default_factory=list, description="Pointer samples.")
    holds: list[Hold] = Field(default_factory=list, description="Holds made elsewhere (clicks).")
    spans: list[SpanHold] = Field(default_factory=list, description="Size gestures made elsewhere.")
    things: list[Thing] = Field(default_factory=list, description="The scene.")
    calls: list[RawCall] = Field(default_factory=list, description="The agent's tool calls.")
    context: Context = Field(default_factory=Context)
    answers: list[Answer] = Field(default_factory=list, description="Replies to an earlier Ask.")
    speech: tuple[float, float] | None = Field(
        default=None, description="Utterance start and end, if the words are missing or cut."
    )
    only: int | None = Field(
        default=None,
        ge=0,
        description="Resolve only this call; the others still take part in matching words to "
        "holds (for robots that act on one call before resolving the next).",
    )


class ResolvedCall(BaseModel):
    """A call with its reference arguments replaced by `Where` or `Size`."""

    model_config = ConfigDict(extra="ignore")

    index: int = Field(description="Its position in Request.calls.")
    name: str
    args: dict[str, Any]

    def where(self, arg: str) -> Where | None:
        v = self.args.get(arg)
        return None if v is None else Where.model_validate(v)

    def size(self, arg: str) -> Size | None:
        v = self.args.get(arg)
        return None if v is None else Size.model_validate(v)


class Response(Wire):
    status: Literal["ok", "ask"]
    calls: list[ResolvedCall] = Field(default_factory=list)
    ask: Ask | None = None
    partial: list[Answer] = Field(
        default_factory=list, description="On ask: what was already resolved, to send back."
    )
    holds: list[Hold] = Field(default_factory=list, description="The holds used, in order.")
    log: list[dict[str, Any]] = Field(default_factory=list, description="Evidence, in order.")


class _Stop(Exception):  # noqa: N818 - control flow
    def __init__(self, ask: Ask) -> None:
        super().__init__(ask.question)
        self.ask = ask


# ---------------------------------------------------------------- resolving


def resolve(request: Request | dict[str, Any], profile: Profile) -> Response:
    """Resolve every reference in the request's calls (plan 4.8)."""
    req = request if isinstance(request, Request) else Request.model_validate(request)
    return _Resolver(req, profile).run()


class _Resolver:
    def __init__(self, req: Request, profile: Profile) -> None:
        self.req = req
        self.profile = profile
        self.rp: ResolverParams = profile.resolver_params()
        self.words = req.words
        self.calls: list[Call] = [parse_call(profile, c.name, c.args) for c in req.calls]
        if req.only is not None and req.only >= len(self.calls):
            raise RequestError(f"only={req.only}, but there are {len(self.calls)} calls")
        holds = list(req.holds)
        spans = list(req.spans) if profile.spans is not None else []
        if req.samples:
            th, ts = track(req.samples, profile)
            holds += th
            spans += ts
        self.holds = holds
        self.spans = spans
        self.avoid = size_holds(holds, spans, self.rp) if spans else set()
        self.answers = {(a.call, a.arg): a for a in req.answers}
        self.things = req.things
        self.used: list[Hold] = []
        self.log: list[dict[str, Any]] = []
        self.partial: list[Answer] = []
        self.picks: dict[int, list[Pick | TimingAsk]] = {}

    # ------------------------------------------------------------------ main loop

    def run(self) -> Response:
        self._match_words()
        out: list[ResolvedCall] = []
        try:
            for i, call in enumerate(self.calls):
                if self.req.only is not None and i != self.req.only:
                    continue
                out.append(self._call(i, call))
        except _Stop as stop:
            return Response(
                status="ask",
                ask=stop.ask,
                partial=self.partial,
                holds=self.used,
                log=self.log,
            )
        return Response(status="ok", calls=out, holds=self.used, log=self.log)

    def _params(self, call: Call) -> list[Param]:
        action = self.profile.action(call.name)
        assert action is not None
        return action.params

    def _match_words(self) -> None:
        """All pointing words of all calls, matched to holds together (F1, k in order)."""
        need: Counter[int] = Counter()
        for i, call in enumerate(self.calls):
            for p in self._params(call):
                if p.type not in ("place", "object") or (i, p.name) in self.answers:
                    continue  # an answered argument keeps its answer
                for idx, n in _deictic_words(call.ref(p.name)).items():
                    if idx < len(self.words):  # a bad index fails its own call, later
                        need[idx] = max(need[idx], n)
        order = [i for i in sorted(need) for _ in range(need[i])]
        if not order:
            return
        picks = assign_holds([self.words[i] for i in order], self.holds, self.rp, self.avoid)
        for i, pick in zip(order, picks, strict=True):
            self.picks.setdefault(i, []).append(pick)

    def _check_index(self, idx: int) -> None:
        if idx >= len(self.words):
            raise RequestError(f"word_index {idx} is past the last word ({len(self.words)} words)")

    def _call(self, i: int, call: Call) -> ResolvedCall:
        args = call.args_dict()
        params = self._params(call)
        resolved: dict[str, Any] = {}
        order = {"object": 0, "size": 1, "place": 2}
        for p in sorted((p for p in params if p.type in order), key=lambda p: order[p.type]):
            value = self._param(i, call, p, resolved)
            if value is not None:
                resolved[p.name] = value
                if isinstance(value, Where):
                    self.partial.append(Answer(call=i, arg=p.name, where=value))
                else:
                    self.partial.append(Answer(call=i, arg=p.name, size=value))
        if call.name == "adjust":
            resolved |= self._adjust(i, call)
        for name, value in resolved.items():
            args[name] = value
        return ResolvedCall(index=i, name=call.name, args=args)

    def _param(self, i: int, call: Call, p: Param, done: dict[str, Any]) -> Where | Size | None:
        ref = call.ref(p.name)
        ans = self.answers.get((i, p.name))
        if ans is not None and ans.hold is None:
            w = self._from_answer(ans, p)
            if p.type == "place" and isinstance(w, Where):
                return self._shape(w, i, call, p, self._size_for(call, done), done)
            return w
        hold = ans.hold if ans is not None else None
        if p.type == "object":
            if ref is None:
                return None
            return self._object(i, p.name, ref, hold, p.allow_point)
        if p.type == "size":
            if ref is None:
                return None
            return self._size(i, p.name, ref)
        return self._place(i, call, p, ref, hold, done)

    def _from_answer(self, ans: Answer, p: Param) -> Where | Size:
        if ans.size is not None:
            return ans.size
        if ans.where is not None:
            return ans.where
        if ans.thing is not None:
            t = next((o for o in self.things if o.id == ans.thing), None)
            if t is None:
                raise RequestError(f"answer names thing '{ans.thing}', which is not in things")
            return self._thing_where(t, {"rule": "answer"})
        raise RequestError("an answer with only a candidate index must re-send its where or thing")

    # ------------------------------------------------------------------ asks

    def _ask(
        self,
        reason: str,
        i: int,
        arg: str,
        variant: str | None = None,
        candidates: Sequence[Where] = (),
        options: Sequence[str] = (),
        word: Word | None = None,
        **fill: Any,
    ) -> _Stop:
        if options and "options" not in fill:
            fill["options"] = " or the ".join(options)
        if len(options) >= 2:
            fill.setdefault("a", options[0])
            fill.setdefault("b", options[1])
        q = self.profile.ask_text(reason, variant, **fill)
        return _Stop(
            Ask(
                reason=reason,  # type: ignore[arg-type]
                question=q,
                call=i,
                arg=arg,
                candidates=list(candidates),
                options=list(options),
                word=word,
            )
        )

    def _timing_ask(self, a: TimingAsk, i: int, arg: str, variant: str | None = None) -> _Stop:
        cands = [self._point(Pick(h, "candidate", a.word), log=False) for h in a.candidates]
        if a.reason == "no_gesture" and a.word is None:
            variant = "pointed"
        return self._ask(a.reason, i, arg, variant, cands, word=a.word)

    # ------------------------------------------------------------------ holds to points

    def _use(self, h: Hold) -> None:
        if not any(h is u for u in self.used):
            self.used.append(h)

    def _point(self, pick: Pick, log: bool = True) -> Where:
        w = hold_where(pick, self.profile.frame, self.rp)
        if log:
            assert isinstance(pick.hold, Hold)
            self._use(pick.hold)
            self.log.append({"where": w.model_dump(mode="json")})
        return w

    def _word_picks(self, idx: int, i: int, arg: str, hold: Hold | None) -> list[Pick]:
        """The picks for one word (two for 'these two'). An answer's hold replaces them."""
        self._check_index(idx)
        word = self.words[idx]
        if hold is not None:
            return [Pick(hold, "answer", word)]
        picks = self.picks.get(idx)
        if not picks:  # not matched up front (should not happen): on its own
            picks = list(assign_holds([word], self.holds, self.rp, self.avoid))
        out = []
        for pk in picks:
            if isinstance(pk, TimingAsk):
                raise self._timing_ask(pk, i, arg)
            out.append(pk)
        return out

    def _pointed(self, i: int, arg: str, hold: Hold | None) -> Where:
        """No pointing word ("a tree" while pointing): the last hold of the utterance."""
        if hold is not None:
            return self._point(Pick(hold, "answer"))
        if self.req.speech is not None:
            t0, t1 = self.req.speech
        elif self.words:
            t0, t1 = self.words[0].t0, self.words[-1].t1
        else:
            raise self._ask("no_gesture", i, arg, "pointed")
        pick = last_hold(t0, t1, self.holds, self.rp, self.avoid)
        if isinstance(pick, TimingAsk):
            raise self._timing_ask(pick, i, arg, "pointed")
        return self._point(pick)

    # ------------------------------------------------------------------ objects

    def _thing_where(self, t: Thing, evidence: dict[str, Any]) -> Where:
        b = t.box
        return Where(
            frame=self.profile.frame,
            kind="box",
            box_mm=(b.x, b.y, b.w, b.h),
            thing=t.id,
            label=t.label,
            source="object_ref",
            evidence=evidence,
        )

    def _object(
        self, i: int, arg: str, ref: Any, hold: Hold | None, allow_point: bool = False
    ) -> Where:
        return self._objects(i, arg, ref, hold, allow_point, 1)[0]

    def _objects(
        self, i: int, arg: str, ref: Any, hold: Hold | None, allow_point: bool, n: int
    ) -> list[Where]:
        """The thing(s) a reference means (F3, F4, F9). n=2 only for 'between these two'."""
        if isinstance(ref, LastObject):
            t = last(self.things, self.req.context.last_thing)
            if t is None:
                raise self._ask("no_object", i, arg, "empty")
            return [self._thing_where(t, {"rule": "last"})]
        if isinstance(ref, LabelObject):
            return [self._labelled(i, arg, ref.label)]
        assert isinstance(ref, DeicticObject)
        picks = self._word_picks(ref.word_index, i, arg, hold)
        if n == 2 and len(picks) < 2:
            raise self._ask("no_gesture", i, arg, word=self.words[ref.word_index])
        out = []
        for pk in picks[:n]:
            point = self._point(pk)
            assert point.mean_mm is not None
            best, cands = object_at(self.things, point.mean_mm, self.profile, ref.label)
            if best is None and cands:
                opts = [c.label or c.id for c in cands]
                wheres = [self._thing_where(c, {"rule": "candidate"}) for c in cands]
                raise self._ask("ambiguous_object", i, arg, None, wheres, opts, pk.word)
            if best is None:
                if allow_point:
                    out.append(point)
                    continue
                raise self._ask("no_object", i, arg, word=pk.word)
            ev = {**point.evidence, "rule": "pointed", "point": list(point.mean_mm)}
            if ref.label:
                ev["label"] = ref.label
            out.append(self._thing_where(best, ev))
        return out

    def _labelled(self, i: int, arg: str, label: str) -> Where:
        hits = find(self.things, label)
        if not hits:
            raise self._ask("unknown_label", i, arg, None, (), labels(self.things)[:4], label=label)
        if len(hits) > 1 and self.profile.scene.label_ties == "ask":
            opts = [h.label or h.id for h in hits]
            wheres = [self._thing_where(h, {"rule": "candidate"}) for h in hits]
            raise self._ask("ambiguous_object", i, arg, None, wheres, opts)
        return self._thing_where(hits[0], {"rule": "label", "label": label})

    # ------------------------------------------------------------------ sizes

    def _size(self, i: int, arg: str, ref: Any) -> Size:
        """A size (F5, F6): a hand gesture, a size word, or the size of another thing."""
        if isinstance(ref, DeicticSize):
            if self.profile.spans is None:
                raise self._ask("unsupported", i, arg)
            self._check_index(ref.word_index)
            word = self.words[ref.word_index]
            pick = pick_span(word, self.spans, self.rp)
            if isinstance(pick, TimingAsk):
                raise self._timing_ask(pick, i, arg)
            span = pick.hold
            assert isinstance(span, SpanHold)
            self.log.append(
                {"size": round(span.span_mm, 2), "size_kind": span.kind, "word": word.text}
            )
            size = Size(
                mm=span.span_mm,
                source="gesture",
                evidence={
                    "word": word.text,
                    "word_t": [round(word.t0, 3), round(word.t1, 3)],
                    "span_t": [round(span.t0, 3), round(span.t1, 3)],
                    "size_kind": span.kind,
                },
            )
        elif isinstance(ref, PresetSize):
            mm = self.profile.sizes.presets.get(ref.preset, self.profile.sizes.default_mm)
            if mm is None:  # the whole area
                a = self.profile.area
                mm = 0.95 * min(a.w, a.h)
            size = Size(mm=mm, source="verbal_preset", evidence={"preset": ref.preset})
        else:
            assert isinstance(ref, RelativeSize)
            obj = self._object(i, arg, ref.same_as, None)
            assert obj.box_mm is not None
            mm = max(obj.box_mm[2], obj.box_mm[3])
            if mm <= 0:
                raise self._ask("no_size", i, arg)
            size = Size(mm=mm, source="object_ref", evidence={"same_as": obj.thing})
        s = self.profile.sizes
        if size.mm < s.min_mm or (s.max_mm is not None and size.mm > s.max_mm):
            raise self._ask("no_size", i, arg, "range")
        return size

    # ------------------------------------------------------------------ places

    def _place(
        self,
        i: int,
        call: Call,
        p: Param,
        ref: Any,
        hold: Hold | None,
        done: dict[str, Any],
    ) -> Where:
        """A place (F1, F2, F7, F8, F12, F17), shaped to what the robot wants."""
        size = self._size_for(call, done)
        w = self._raw_place(i, p.name, ref, hold, size)
        w = self._shape(w, i, call, p, size, done)
        return self._outside(w, i, p.name)

    def _size_for(self, call: Call, done: dict[str, Any]) -> float:
        for q in self._params(call):
            if q.type == "size" and isinstance(done.get(q.name), Size):
                return done[q.name].mm
        return self.profile.sizes.default_mm

    def _geo(self, c: Sequence[float], evidence: dict[str, Any], z: float | None = None) -> Where:
        return Where(
            frame=self.profile.frame,
            kind="point",
            mean_mm=(round(c[0], 2), round(c[1], 2)),
            z_mm=z,
            source="geometry",
            evidence=evidence,
        )

    def _raw_place(self, i: int, arg: str, ref: Any, hold: Hold | None, size: float) -> Where:
        if ref is None or isinstance(ref, PointedPlace):
            return self._pointed(i, arg, hold)
        if isinstance(ref, DeicticPlace):
            (pk,) = self._word_picks(ref.word_index, i, arg, hold)[:1]
            return self._point(pk)
        if isinstance(ref, RelationPlace):
            return self._relation(i, arg, ref, hold, size)
        if isinstance(ref, NamedPlace):
            c = named(ref.named, size, self.profile)
            if c is None:
                raise self._ask("unknown_place", i, arg, name=ref.named)
            return self._geo(c, {"named": ref.named})
        if isinstance(ref, EmptySpace):
            res = empty_space(self.things, size, self.profile)
            if res is None:
                raise self._ask("no_room", i, arg)
            c, side = res
            return self._geo(c, {"empty_space": True, "side_mm": round(side, 1), "size_mm": side})
        assert isinstance(ref, TwoPlaces)
        return self._two(i, arg, ref, hold, size)

    def _relation(
        self, i: int, arg: str, ref: RelationPlace, hold: Hold | None, size: float
    ) -> Where:
        target = ref.target
        if ref.relation == "between" and ref.target2 is None:
            assert isinstance(target, DeicticObject)  # "between these two"
            a, b = self._objects(i, arg, target, hold, False, 2)
        else:
            a = self._object(i, arg, target, hold)
            b = self._object(i, arg, ref.target2, hold) if ref.target2 is not None else None
        ta, tb = self._thing(a), self._thing(b) if b is not None else None
        new_size = None
        if ref.relation == "inside":
            new_size = min(size, inside_size(ta, self.profile))
        try:
            spot = relation(
                ref.relation, ta, new_size or size, self.profile, self.things, target2=tb
            )
        except Unsupported:
            raise self._ask("unsupported", i, arg) from None
        ev: dict[str, Any] = {"relation": ref.relation, "target": ta.id}
        if tb is not None:
            ev["target2"] = tb.id
        if new_size is not None:
            ev["size_mm"] = new_size
        return self._geo(spot.center, ev, spot.z_mm)

    def _thing(self, w: Where) -> Thing:
        return next(t for t in self.things if t.id == w.thing)

    def _two(self, i: int, arg: str, ref: TwoPlaces, hold: Hold | None, size: float) -> Where:
        """'From here ... to here': two corners (F7)."""
        ends = []
        for corner in (ref.from_, ref.to):
            h = hold if hold is not None and self._corner_failed(corner) else None
            ends.append(self._raw_place(i, arg, corner, h, size))
        (a, b) = ends
        assert a.mean_mm is not None and b.mean_mm is not None
        (ax, ay), (bx, by) = a.mean_mm, b.mean_mm
        x0, y0 = min(ax, bx), min(ay, by)
        box = (round(x0, 2), round(y0, 2), round(abs(bx - ax), 2), round(abs(by - ay), 2))
        cov = tuple(
            tuple((a.cov_mm2[r][c] + b.cov_mm2[r][c]) / 4 for c in range(2)) for r in range(2)
        )
        rules = {a.evidence.get("rule"), b.evidence.get("rule")} - {None}
        ev: dict[str, Any] = {
            "rule": rules.pop() if len(rules) == 1 else "corners",
            "corners": [list(a.mean_mm), list(b.mean_mm)],
            "from": a.evidence,
            "to": b.evidence,
        }
        both_gesture = a.source == b.source == "gesture"
        return Where(
            frame=self.profile.frame,
            kind="box",
            box_mm=box,
            cov_mm2=cov,  # type: ignore[arg-type]
            source="gesture" if both_gesture else "geometry",
            confidence=min(a.confidence, b.confidence),
            evidence=ev,
        )

    def _corner_failed(self, corner: Any) -> bool:
        """True if this corner's pointing word found no usable hold (an answer fills it)."""
        if isinstance(corner, DeicticPlace):
            picks = self.picks.get(corner.word_index, [])
            return not picks or any(isinstance(pk, TimingAsk) for pk in picks)
        return isinstance(corner, PointedPlace)

    def _shape(
        self, w: Where, i: int, call: Call, p: Param, size: float, done: dict[str, Any]
    ) -> Where:
        """Turn the place into the shape the robot wants (point, box or path)."""
        want = p.want
        if want == "point":
            if w.kind == "point":
                return w
            assert w.mean_mm is not None
            return w.model_copy(update={"kind": "point", "box_mm": None, "path_mm": None})
        if want == "path":
            if w.kind == "path":
                return w
            if w.kind == "box" and "corners" in w.evidence:
                a, b = w.evidence["corners"]
                return w.model_copy(
                    update={"kind": "path", "box_mm": None, "path_mm": [tuple(a), tuple(b)]}
                )
            raise self._ask("unsupported", i, p.name)
        # want box
        if w.kind == "box":
            return w
        assert w.mean_mm is not None
        has_size = any(q.type == "size" for q in self._params(call))
        frm = p.size_from or ("size_param" if has_size else "default")
        ev = dict(w.evidence)
        if frm == "object":
            obj = self._call_object(done)
            if obj is None or obj.box_mm is None:
                frm = "default"
            else:
                bw, bh = obj.box_mm[2], obj.box_mm[3]
                ev |= {"box_from": "object", "size_mm": [bw, bh]}
        if frm != "object":
            s = ev.get("size_mm")
            if not isinstance(s, (int, float)):
                s = size if frm == "size_param" else self.profile.sizes.default_mm
            bw = bh = float(s)
            ev |= {"box_from": "size" if frm == "size_param" else "default", "size_mm": s}
        cx, cy = w.mean_mm
        box = (round(cx - bw / 2, 2), round(cy - bh / 2, 2), bw, bh)
        return w.model_copy(update={"kind": "box", "box_mm": box, "evidence": ev})

    def _call_object(self, done: dict[str, Any]) -> Where | None:
        for v in done.values():
            if isinstance(v, Where) and v.thing is not None:
                return v
        return None

    def _outside(self, w: Where, i: int, arg: str) -> Where:
        """Outside the work area: keep it and say so, or ask. Never move it (rule 1.3.4)."""
        assert w.mean_mm is not None
        a = self.profile.area
        x, y = w.mean_mm
        if a.x <= x <= a.x2 and a.y <= y <= a.y2:
            return w
        if self.profile.outside == "ask":
            raise self._ask("outside_area", i, arg, candidates=[w])
        return w.model_copy(update={"evidence": {**w.evidence, "outside": True}})

    # ------------------------------------------------------------------ adjust

    def _adjust(self, i: int, call: Call) -> dict[str, Any]:
        prev = self.req.context.previous
        if not prev:
            raise self._ask("no_previous", i, "place")
        dx, dy, scale = call.ref("dx"), call.ref("dy"), call.ref("scale")
        out: dict[str, Any] = {}
        for key, value in prev.items():
            if isinstance(value, Where):
                out[key] = adjust(value, dx, dy, scale, self.profile)
            elif isinstance(value, Size):
                out[key] = adjust_size(value, scale)
        return out


def _deictic_words(ref: Any) -> Counter[int]:
    """Word indices of the pointing words inside a place or object reference, each with the
    number of holds it needs (2 for 'these two')."""
    out: Counter[int] = Counter()
    if isinstance(ref, DeicticObject):
        out[ref.word_index] = max(out[ref.word_index], ref.count or 1)
    elif isinstance(ref, DeicticPlace):
        out[ref.word_index] = max(out[ref.word_index], 1)
    elif isinstance(ref, BaseModel):
        for f in type(ref).model_fields:
            for idx, n in _deictic_words(getattr(ref, f)).items():
                out[idx] = max(out[idx], n)
    return out


# ---------------------------------------------------------------- adjust


def adjust(
    where: Where, dx: float = 0, dy: float = 0, scale: float = 1.0, profile: Profile | None = None
) -> Where:
    """A small fix to a previewed place (F10): dx and dy in steps along the viewer's right and
    away directions (one step = the profile's `adjust.step_mm`), scale about the centre."""
    if profile is None:
        raise ValueError("adjust needs the robot's profile (step size and directions)")
    step = profile.adjust.step_mm
    r, a = profile.viewer.right, profile.viewer.away
    ox = (dx * r[0] + dy * a[0]) * step
    oy = (dx * r[1] + dy * a[1]) * step
    assert where.mean_mm is not None
    cx, cy = where.mean_mm
    mean = (cx + ox, cy + oy)  # exact: a robot may move its own drawing by the difference
    box = None
    if where.box_mm is not None:
        x, y, w, h = where.box_mm
        nw, nh = w * scale, h * scale
        bx, by = x + w / 2 + ox, y + h / 2 + oy
        box = (bx - nw / 2, by - nh / 2, nw, nh)
    path = None
    if where.path_mm is not None:
        path = [
            (cx + (px - cx) * scale + ox, cy + (py - cy) * scale + oy) for px, py in where.path_mm
        ]
    ev: dict[str, Any] = {
        "dx": dx,
        "dy": dy,
        "scale": scale,
        "step_mm": step,
        "offset_mm": [ox, oy],  # what to move the robot's own copy by
        "previous": where.model_dump(mode="json", exclude={"version"}),
    }
    area = profile.area
    if not (area.x <= mean[0] <= area.x2 and area.y <= mean[1] <= area.y2):
        ev["outside"] = True
    return where.model_copy(
        update={
            "mean_mm": mean,
            "box_mm": box,
            "path_mm": path,
            "source": "adjust",
            "evidence": ev,
        }
    )


def adjust_size(size: Size, scale: float) -> Size:
    """'Bigger', 'smaller': the size times `scale` (F10)."""
    return Size(
        mm=size.mm * scale,
        source="adjust",
        confidence=size.confidence,
        evidence={"scale": scale, "previous": size.model_dump(mode="json", exclude={"version"})},
    )

"use client";

/**
 * The two animations, and the reduced-motion substitute for both.
 *
 * PlumeRank has exactly two animations: the FLIP rank reorder and a budget slot
 * filling. Everything else is instant. That is a mechanism decision rather than a
 * taste one -- motion in this product means "look here", and a UI that animates
 * everything cannot use motion to mean anything -- so the code that implements them
 * lives in one file where the count is checkable by reading it.
 */

import { useEffect, useLayoutEffect, useRef, useState } from "react";

/** useLayoutEffect warns during a server render. This export is a static site, so
 *  there IS no server render at request time -- but `next build` still prerenders the
 *  shell once, and a build-time warning in the deploy log is a real cost for nothing. */
const useIsomorphicLayoutEffect =
  typeof window === "undefined" ? useEffect : useLayoutEffect;

export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(mq.matches);
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return reduced;
}

/**
 * FLIP: First, Last, Invert, Play.
 *
 * Measure every row before React commits the new order, measure again after, then put
 * each row back where it was with a transform and release it. The browser animates a
 * compositor-only property, so a 40-row reorder costs no layout per frame -- which is
 * what makes this affordable on a box that is simultaneously running 40 background
 * models.
 *
 * Under `prefers-reduced-motion` the transform is never applied. Instead a row that
 * moved gets a one-frame --signal outline: the point of the animation is the "look
 * here" signal, so removing the motion without substituting the signal would delete a
 * channel rather than respect a preference.
 *
 * @param keys row identities in their current order -- changing this array is what
 *             triggers a measurement.
 */
/** The longest reorder the FLIP animates. See the comment at the call site. */
const MAX_FLIP_PX = 320;

export function useFlip(keys: string[], reduced: boolean) {
  const container = useRef<HTMLDivElement | null>(null);
  const previous = useRef<Map<string, number>>(new Map());

  useIsomorphicLayoutEffect(() => {
    const root = container.current;
    if (!root) return;

    const nodes = Array.from(
      root.querySelectorAll<HTMLElement>("[data-flip-key]"),
    );
    const next = new Map<string, number>();

    // READ every position first, then WRITE. Interleaving the two -- a
    // getBoundingClientRect() after the previous row's transform write -- forced one
    // synchronous layout per moved row on every tick (Lighthouse on the live box: ~0.9 s
    // of blocking time on a throttled phone, "Style & Layout" 2.8 s of 10 s). Reading
    // before any write gives the same tops, because nothing in this tick has moved yet.
    const measured: Array<[HTMLElement, string, number]> = [];
    for (const node of nodes) {
      const key = node.dataset.flipKey;
      if (!key) continue;
      const top = node.getBoundingClientRect().top;
      next.set(key, top);
      measured.push([node, key, top]);
    }

    const inverted: HTMLElement[] = [];
    for (const [node, key, top] of measured) {
      const before = previous.current.get(key);
      if (before === undefined || Math.abs(before - top) < 1) continue;

      // A move longer than this is not animated, and the rule earns its place twice.
      // The reason it exists: FLIP is here so the eye can FOLLOW one tile climbing, and
      // a jump of twenty rows cannot be followed -- animating it produces 220 ms of
      // motion that means nothing, which is the exact failure the two-animation budget
      // was drawn to prevent. The bug it fixes: the watch list scrolls inside a 420 px
      // rail, so a row inverted several hundred pixels is transformed outside the
      // container and CLIPPED, leaving a hole in the list for the length of the
      // transition. Long moves take the same one-frame outline reduced motion gets.
      const far = Math.abs(before - top) > MAX_FLIP_PX;

      if (reduced || far) {
        node.classList.add("changed");
        // Two frames: one for the class to paint, one to take it away. A single rAF
        // can be coalesced into the same paint as the class addition, and the outline
        // then never appears at all -- which is the failure mode that looks exactly
        // like "reduced motion works".
        requestAnimationFrame(() =>
          requestAnimationFrame(() => node.classList.remove("changed")),
        );
        continue;
      }

      node.classList.remove("flip-move");
      node.style.transform = `translateY(${before - top}px)`;
      inverted.push(node);
    }

    if (inverted.length > 0) {
      // ONE forced layout commits every inverted position before the transition class
      // goes back on. Without it the browser coalesces both style changes into one
      // recalculation and there is nothing to animate FROM.
      void root.offsetHeight;
      for (const node of inverted) {
        node.classList.add("flip-move");
        node.style.transform = "";
      }
    }

    previous.current = next;
  }, [keys.join("|"), reduced]);

  return container;
}

"use client";

import { useEffect, useRef, useState } from "react";

/**
 * An <img> that keeps showing the last frame that finished loading until a newer one has
 * decoded. Evidence crops change URL on every 1 s tick; a plain <img src> aborted the crop
 * still in flight on every change, so on a real connection the top rows sat as empty
 * boxes nearly all the time (live QA, 2026-10-12: 0 of 8 loaded in 23 of 24 samples).
 *
 * Unlike the wall's TileImage, a load that finishes after a newer URL was requested is
 * still shown if nothing newer has finished: a crop one tick old beats an empty box.
 */
export function BufferedImg({ src, className }: { src: string; className?: string }) {
  const [shown, setShown] = useState<string | null>(null);
  const requested = useRef(0);
  const displayed = useRef(0);

  useEffect(() => {
    const ticket = ++requested.current;
    const next = new Image();
    next.decoding = "async";
    next.src = src;
    next
      .decode()
      .then(() => {
        if (ticket > displayed.current) {
          displayed.current = ticket;
          setShown(src);
        }
      })
      .catch(() => {
        /* keep the previous crop */
      });
  }, [src]);

  return shown ? (
    <img className={className} src={shown} alt="" decoding="async" />
  ) : (
    <span className={className} aria-hidden="true" />
  );
}

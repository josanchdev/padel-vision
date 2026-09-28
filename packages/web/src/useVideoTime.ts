// The video's current time as React state: the single clock every panel of the
// point view follows. Frame-accurate while playing (requestAnimationFrame),
// and updated on seeks while paused.
import { useEffect, useState } from "react";

export function useVideoTime(video: HTMLVideoElement | null): number {
  const [time, setTime] = useState(0);

  useEffect(() => {
    if (!video) return;
    let frame = 0;
    const tick = () => {
      setTime(video.currentTime);
      if (!video.paused) frame = requestAnimationFrame(tick);
    };
    const onPlay = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(tick);
    };
    const sync = () => setTime(video.currentTime);
    video.addEventListener("play", onPlay);
    video.addEventListener("seeked", sync);
    video.addEventListener("timeupdate", sync);
    video.addEventListener("loadedmetadata", sync);
    return () => {
      cancelAnimationFrame(frame);
      video.removeEventListener("play", onPlay);
      video.removeEventListener("seeked", sync);
      video.removeEventListener("timeupdate", sync);
      video.removeEventListener("loadedmetadata", sync);
    };
  }, [video]);

  return time;
}

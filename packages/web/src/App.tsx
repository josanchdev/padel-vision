// Two screens, chosen by the URL hash: the home grid (#/) and one point (#/p/<id>).
// A hash router needs no server rules, so the build runs from any static host
// or straight from `vite preview`.
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useState } from "react";

import { Home } from "./pages/Home.tsx";
import { PointView } from "./pages/PointView.tsx";

type Route = { page: "home" } | { page: "point"; id: string };

function parse(hash: string): Route {
  const match = hash.match(/^#\/p\/([\w-]+)$/);
  return match ? { page: "point", id: match[1] } : { page: "home" };
}

export function App() {
  const [route, setRoute] = useState<Route>(() => parse(window.location.hash));

  useEffect(() => {
    const onChange = () => {
      setRoute(parse(window.location.hash));
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  const key = route.page === "point" ? `p-${route.id}` : "home";
  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={key}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
      >
        {route.page === "point" ? <PointView id={route.id} /> : <Home />}
      </motion.div>
    </AnimatePresence>
  );
}

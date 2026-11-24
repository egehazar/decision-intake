/* Quick Benefits Check, embedded. Put this where the screener should appear:
     <script src="https://decision-intake.vercel.app/embed.js" async></script>
   Optional: data-lang="es" for Spanish (default: the visitor's browser language), and data-theme="light" or
   "dark" to match your page (default: the visitor's system setting).
   It adds one sandboxed iframe that grows with its content. Nothing is stored on your site or ours. */
(function () {
  "use strict";
  var s = document.currentScript;
  if (!s) return;
  var origin = new URL(s.src).origin;
  var lang = s.getAttribute("data-lang"), theme = s.getAttribute("data-theme");
  var opts = [lang === "es" || lang === "en" ? lang : "", theme === "light" || theme === "dark" ? theme : ""]
    .filter(Boolean).join(",");
  var f = document.createElement("iframe");
  f.src = origin + "/embed" + (opts ? "#" + opts : "");
  f.title = lang === "es" ? "Revisión rápida de beneficios" : "Quick Benefits Check";
  f.setAttribute("sandbox", "allow-scripts allow-same-origin allow-forms allow-popups allow-popups-to-escape-sandbox allow-modals");
  f.setAttribute("referrerpolicy", "no-referrer");
  f.style.cssText = "display:block;width:100%;max-width:760px;height:640px;border:0;border-radius:12px;color-scheme:normal";
  window.addEventListener("message", function (e) {
    if (e.origin !== origin || e.source !== f.contentWindow || !e.data || e.data.type !== "decisionintake:height") return;
    var h = Number(e.data.height);
    if (h > 200 && h < 20000) f.style.height = Math.ceil(h) + "px";
    if (e.data.top) f.scrollIntoView({block: "start", behavior: "smooth"});  // a new screen: show its top
  });
  s.parentNode.insertBefore(f, s.nextSibling);
})();

// The chat page's Markdown renderer (llvq-runtime/src/chat.html) against
// hostile and odd model output: no script, no event handler, no javascript:
// link, and no throw. A throw turns a whole answer into an error on the page.
//
//   node ops/llvq_chat_markdown_check.js llvq-runtime/src/chat.html
//
// Not in `cargo test`: CI has no Node. Run it after any change to the page's
// renderer (docs/runtime-audit-2026-11.md).
const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8");
const a = html.indexOf("function escapeHtml"), b = html.indexOf("async function copyText");
const ICON = { copy: "" };
eval(html.slice(a, b) + "; globalThis.markdown = markdown;");
const cases = ["plain **bold** `code`", "a\u00000\u0000b", "x `y` \u00001\u0000", "<script>alert(1)</script>", "[t](javascript:alert(1))", '[t](https://e.x/"onmouseover=1)'];
let bad = 0;
for (const c of cases) {
  try { const out = markdown(c); console.log(JSON.stringify(c), "->", out); if (/<script|onmouseover="|href="javascript/.test(out)) { bad++; console.log("  UNSAFE"); } }
  catch (e) { bad++; console.log(JSON.stringify(c), "-> THROWS", e.message); }
}
process.exit(bad ? 1 : 0);

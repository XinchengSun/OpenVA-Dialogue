"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.resolve(__dirname, "../static/index.html"), "utf8");
const source = html.match(/function requireServicePage\(\) \{[\s\S]*?\n\}/)[0] + '\n' +
  html.match(/function wsUrl\(path\) \{[\s\S]*?\n\}/)[0];
const cases = [
  ["http://localhost:7860/", "ws://localhost:7860/"],
  ["https://example.test/proxy/18786/", "wss://example.test/proxy/18786/"],
  ["https://example.test/workspace/a/proxy/18786/?token=not-forwarded", "wss://example.test/workspace/a/proxy/18786/"],
  ["https://example.test/openva/index.html", "wss://example.test/openva/"],
  ["http://[::1]:7860/", "ws://[::1]:7860/"],
];
for (const [href, prefix] of cases) {
  const location = new URL(href);
  const context = vm.createContext({URL, location});
  vm.runInContext(source, context);
  for (const endpoint of ["logs", "media", "mic"]) {
    context.endpoint = "/ws/" + endpoint;
    assert.equal(vm.runInContext("wsUrl(endpoint)", context), prefix + "ws/" + endpoint);
  }
}
for (const href of ['file:///D:/OpenVA/static/index.html', 'data:text/html,hello']) {
  const context = vm.createContext({URL, location:new URL(href)});
  vm.runInContext(source, context);
  assert.throws(() => vm.runInContext("wsUrl('/ws/mic')", context), /服务网址/);
}
console.log("WS_PROXY_PATH_OK 15 route cases and 2 unsupported page cases");

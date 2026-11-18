import { existsSync, readFileSync, statSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";

const PACKAGE = "@github/copilot-sdk";
const VERSION = "1.0.28";

function packageMetadata(entry) {
  let directory = path.dirname(entry);
  while (true) {
    const file = path.join(directory, "package.json");
    if (existsSync(file)) {
      const metadata = JSON.parse(readFileSync(file, "utf8"));
      if (metadata.name === PACKAGE) return { ...metadata, root: directory };
    }
    const parent = path.dirname(directory);
    if (parent === directory) throw new Error(`Cannot identify the package owning ${entry}`);
    directory = parent;
  }
}

export function loadCopilotSdk({
  projectDirectory = process.cwd(),
  explicitRoot = process.env.SCIENCE_HARNESS_SDK_ROOT,
} = {}) {
  let anchors;
  if (explicitRoot !== undefined) {
    if (typeof explicitRoot !== "string" || !explicitRoot.trim() || !existsSync(explicitRoot)
        || !statSync(explicitRoot).isDirectory()
        || !existsSync(path.join(explicitRoot, "package.json"))) {
      throw new Error("SCIENCE_HARNESS_SDK_ROOT must name an npm project containing package.json");
    }
    anchors = [createRequire(path.join(path.resolve(explicitRoot), "package.json"))];
  } else {
    anchors = [
      createRequire(path.join(path.resolve(projectDirectory), "package.json")),
      createRequire(import.meta.url),
    ];
  }
  for (const require of anchors) {
    let entry;
    try {
      entry = require.resolve(PACKAGE);
    } catch (error) {
      if (error.code === "MODULE_NOT_FOUND") continue;
      throw error;
    }
    const metadata = packageMetadata(entry);
    if (metadata.version !== VERSION) {
      throw new Error(`Expected ${PACKAGE}@${VERSION}, found ${metadata.version} at ${metadata.root}`);
    }
    // Load only after resolving and validating the selected installation. Missing
    // transitive dependencies must fail, not silently select a different SDK.
    const sdk = require(PACKAGE);
    if (typeof sdk.CopilotClient !== "function" || typeof sdk.RuntimeConnection?.forStdio !== "function") {
      throw new Error("The selected Copilot SDK lacks the required runtime API");
    }
    return { ...sdk, sdkResolution: { version: metadata.version, packageRoot: metadata.root } };
  }
  throw new Error(`Cannot resolve ${PACKAGE}@${VERSION}; run npm ci in the operator project `
    + "or set SCIENCE_HARNESS_SDK_ROOT to its directory");
}

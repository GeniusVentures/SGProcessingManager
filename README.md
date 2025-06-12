## 🧪 Schema-to-C++ Header Generation

This project uses [QuickType](https://quicktype.io) to generate C++ headers from the `gnus-processing-schema.json` file. This ensures your model classes stay in sync with the schema specification.

### 📦 Install QuickType

First, install QuickType globally using `npm`:

```bash
npm install -g quicktype
```

> You must have [Node.js](https://nodejs.org/) installed. This also provides `npm`.

---

### ⚙️ Generate C++ Headers

To regenerate headers manually from the schema:

```bash
quicktype \
  --src-lang schema \
  --lang cpp \
  --top-level SGNSProcessing \
  --code-format with-getter-setter \
  --const-style west-const \
  --namespace sgns \
  --type-style pascal-case \
  --member-style underscore-case \
  --boost \
  --source-style multi-source \
  --include-location global-include \
  --out generated/SGNSProcMain.hpp \
  gnus-processing-schema.json
```

This command:
- Parses `gnus-processing-schema.json`
- Generates multiple C++ header files into the `generated/` directory
- Applies naming conventions and Boost support
- Uses `SGNSProcessing` as the root class name in the `sgns` namespace

---

### 💡 Tips

- Only edit the `gnus-processing-schema.json` file — the headers are regenerated from it.
- A GitHub Action will automatically create a PR with regenerated headers on schema change.
- If you want to manually verify changes before pushing, inspect `git diff` after running the command.

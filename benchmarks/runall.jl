using Weave

FILES = [
    "multi-language-wrappers.jmd",
]

for file in FILES
    weave(file; doctype="github", out_path="../docs/src/benchmarks/", cache=:all)
end

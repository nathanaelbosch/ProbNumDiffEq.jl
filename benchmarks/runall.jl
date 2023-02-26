using Weave

FILES = [
    "lotkavolterra.jmd",
    "multi-language-wrappers.jmd",
]

for file in FILES
    @info "Weave file" file
    weave(
        file;
        doctype="github",
        out_path="../docs/src/benchmarks/",
        fig_ext=".svg",
    )
end

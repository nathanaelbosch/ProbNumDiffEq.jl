using Weave
ENV["GKSwstype"]="nul"

FILES = [
    "lotkavolterra.jmd",
    "vanderpol.jmd",
    "rober.jmd",
    "pleiades.jmd",
    "multi-language-wrappers.jmd",
]

filedir = @__DIR__
for file in FILES
    @info "Weave file" file
    weave(
        file;
        doctype="github",
        out_path=joinpath(filedir, "../docs/src/benchmarks/"),
        fig_ext=".svg",
    )
end

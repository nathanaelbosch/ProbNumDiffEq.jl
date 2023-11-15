using Weave
ENV["GKSwstype"] = "nul"
set_chunk_defaults!(
    :fig_width => 9,
    :fig_height => 5,
)

FILES = [
    "lotkavolterra.jmd",
    "hodgkinhuxley.jmd",
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

restore_chunk_defaults!()
delete!(ENV, "GKSwstype")
nothing

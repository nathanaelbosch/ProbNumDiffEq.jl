using Weave

Weave.register_format!(
    "custom",
    Weave.GitHubMarkdown(
        codestart="""
        ```@raw html
        <details><summary>Code:</summary>
        ```
        ```julia""",
        codeend="""
        ```
        ```@raw html
        </details>
        ```
        """,
    ),
)

ENV["GKSwstype"] = "nul"

set_chunk_defaults!(
    :fig_width => 9,
    :fig_height => 5,
)

FILES = [
    "lotkavolterra.jmd",
    "hodgkinhuxley.jmd",
    "vanderpol.jmd",
    "pleiades.jmd",
    "rober.jmd",
    "orego.jmd",
    "multi-language-wrappers.jmd",
]

filedir = @__DIR__
for file in FILES
    @info "Weave file" file
    weave(
        file;
        doctype="custom",
        out_path=joinpath(filedir, "../docs/src/benchmarks/"),
        fig_ext=".svg",
    )
end

restore_chunk_defaults!()
delete!(ENV, "GKSwstype")
nothing

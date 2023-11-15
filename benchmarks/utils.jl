using InteractiveUtils, Pkg, Markdown

function appendix()
    vinfo = sprint(InteractiveUtils.versioninfo)
    proj = sprint(io -> Pkg.status(io=io))
    mani = sprint(io -> Pkg.status(io=io, mode=Pkg.PKGMODE_MANIFEST))

    display(Markdown.parse("""
    ```@raw html
    <details><summary>Computer information:</summary>
    ```
    ```julia
    using InteractiveUtils
    InteractiveUtils.versioninfo()
    ```
    ```
    $(chomp(vinfo))
    ```
    ```@raw html
    </details>
    ```

    ```@raw html
    <details><summary>Package information:</summary>
    ```
    ```julia
    using Pkg
    Pkg.status()
    ```
    ```
    $(chomp(proj))
    ```
    ```@raw html
    </details>
    ```

    ```@raw html
    <details><summary>Full manifest:</summary>
    ```
    ```julia
    Pkg.status(mode=Pkg.PKGMODE_MANIFEST)
    ```
    ```
    $(chomp(mani))
    ```
    ```@raw html
    </details>
    ```
    """))
end

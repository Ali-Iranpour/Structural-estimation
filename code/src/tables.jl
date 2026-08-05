# =============================================================================
# tables.jl — LaTeX table output (M1).
#
# Emits `threeparttable` + `booktabs` tables matching the house format used in
# Redistribution_and_Human_Capital/{Tables,outcomes}:
#
#   \begin{table}[H] \centering \begin{threeparttable}
#     \caption{...} \label{tab:...}
#     \begin{tabular}{...}
#       \toprule\toprule  ...  \midrule  ...  \bottomrule\bottomrule
#     \end{tabular}
#     \begin{tablenotes}\footnotesize \item[a] ... \item \textit{Note:} ...
#   \end{threeparttable} \end{table}
#
# Every table is written to output/tables/<name>.tex and is \input-able directly into
# the paper. `build_tables_pdf()` compiles them all into one reviewable PDF.
#
# Reproducibility: every writer also emits <name>.meta.toml recording the git commit,
# timestamp, Julia/package versions and the parameters behind the numbers, via
# manifest.jl. Requires paths.jl and manifest.jl.
# =============================================================================

"""
    fmt_num(x; digits = 2, thousands = false)

Format a number for a LaTeX cell. `--` for non-finite, optional thousands separators.
"""
function fmt_num(x; digits::Int = 2, thousands::Bool = false)
    (x === nothing || (x isa Real && !isfinite(x))) && return "--"
    # digits = 0 must give "1712", not "1712.0": Float64 printing always keeps a decimal.
    s = digits == 0 ? string(round(Int, float(x))) : string(round(float(x), digits = digits))
    # round() drops trailing zeros; pad so a column lines up
    if digits > 0
        parts = split(s, ".")
        frac = length(parts) == 2 ? parts[2] : ""
        s = parts[1] * "." * rpad(frac, digits, '0')
    end
    if thousands
        neg = startswith(s, "-"); body = neg ? s[2:end] : s
        ip, fp = split(body, "."; limit = 2)[1], (occursin(".", body) ? split(body, "."; limit = 2)[2] : "")
        grouped = reverse(join([reverse(ip)[i:min(i+2, end)] for i in 1:3:length(ip)], ","))
        s = (neg ? "-" : "") * grouped * (isempty(fp) ? "" : "." * fp)
    end
    return s
end

fmt_int(x) = x === nothing ? "--" : fmt_num(x; digits = 0, thousands = true)

"""
    latex_escape(s)

Escape the characters that break a LaTeX cell. Leaves math mode (`\$...\$`) alone.
"""
function latex_escape(s::AbstractString)
    occursin('\$', s) && return s              # assume the caller wrote deliberate math
    for (a, b) in ("&" => "\\&", "%" => "\\%", "_" => "\\_", "#" => "\\#")
        s = replace(s, a => b)
    end
    return s
end

"""
    write_table(name; caption, label, colspec, header, rows, notes=String[],
                tnotes=Pair{String,String}[], midrules=Int[], small=false, params...)

Write `output/tables/<name>.tex` in the house format, plus `<name>.meta.toml` provenance.

* `colspec`  — the `tabular` column spec, e.g. `"p{5.2cm} p{2.8cm} p{2.8cm}"`
* `header`   — column headings; wrapped in `\\textbf{}` automatically
* `rows`     — vector of vectors of pre-formatted strings
* `midrules` — row indices after which to insert a `\\midrule` (e.g. before a totals row)
* `tnotes`   — `("a" => "text")` pairs rendered as `\\item[a]`
* `notes`    — plain notes; the first is prefixed `\\textit{Note:}`
* `params…`  — recorded in the provenance file, not printed

Returns the path written.
"""
function write_table(name::AbstractString;
                     caption::AbstractString,
                     label::AbstractString,
                     colspec::AbstractString,
                     header::Vector{<:AbstractString},
                     rows::Vector,
                     notes::Vector{<:AbstractString} = String[],
                     tnotes::Vector{<:Pair} = Pair{String,String}[],
                     midrules::Vector{Int} = Int[],
                     small::Bool = false,
                     params...)

    io = IOBuffer()
    println(io, "\\begin{table}[H]")
    println(io, "    \\centering")
    println(io, "    \\begin{threeparttable}")
    println(io, "    \\caption{", caption, "}")
    println(io, "    \\label{tab:", label, "}")
    small && println(io, "    \\small")
    println(io, "    \\begin{tabular}{", colspec, "}")
    println(io, "        \\toprule\\toprule")
    println(io, "        ", join(["\\textbf{" * h * "}" for h in header], " & "), " \\\\")
    println(io, "        \\midrule")
    for (i, r) in enumerate(rows)
        println(io, "        ", join(string.(r), " & "), " \\\\")
        i in midrules && println(io, "        \\midrule")
    end
    println(io, "        \\bottomrule\\bottomrule")
    println(io, "    \\end{tabular}")
    if !isempty(notes) || !isempty(tnotes)
        println(io, "    \\begin{tablenotes}")
        println(io, "        \\footnotesize")
        for (k, v) in tnotes
            println(io, "        \\item[", k, "] ", v)
        end
        for (i, n) in enumerate(notes)
            println(io, "        \\item ", i == 1 ? "\\textit{Note:} " : "", n)
        end
        println(io, "    \\end{tablenotes}")
    end
    println(io, "    \\end{threeparttable}")
    println(io, "\\end{table}")

    dir  = tabpath()
    path = joinpath(dir, name * ".tex")
    write(path, String(take!(io)))

    # provenance alongside every table
    try
        mpath = joinpath(dir, name * ".meta.toml")
        buf = IOBuffer()
        println(buf, "# Provenance for ", name, ".tex — written automatically.")
        println(buf, "[table]")
        println(buf, "name       = ", _toml_val(name))
        println(buf, "caption    = ", _toml_val(caption))
        println(buf, "label      = ", _toml_val("tab:" * label))
        println(buf, "n_rows     = ", length(rows))
        println(buf, "timestamp  = ", _toml_val(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")))
        println(buf, "git_commit = ", _toml_val(git_sha()))
        println(buf, "julia      = ", _toml_val(string(VERSION)))
        if !isempty(params)
            println(buf); println(buf, "[parameters]")
            for (k, v) in pairs(params)
                println(buf, rpad(string(k), 20), "= ", _toml_val(v))
            end
        end
        write(mpath, String(take!(buf)))
    catch err
        @warn "Could not write table provenance" name err
    end

    @info "Wrote table" path
    return path
end

# -----------------------------------------------------------------------------
# Model-specific writers
# -----------------------------------------------------------------------------

"""
    table_college_work(path_choice, name; caption, label, note)

College vs. work counts and shares, in the format of `base_college_work_choice.tex`.
"""
function table_college_work(path_choice, name::AbstractString;
                            caption::AbstractString = "College vs.\\ Work Decisions",
                            label::AbstractString = "college_work_choice",
                            note::AbstractString = "Table reports the number and percentage of individuals choosing each post-family-stage path.",
                            params...)
    n   = length(path_choice)
    ncl = count(==(:college), path_choice)
    nwk = n - ncl
    rows = [["College", fmt_int(ncl), fmt_num(100ncl/n; digits = 1)],
            ["Work",    fmt_int(nwk), fmt_num(100nwk/n; digits = 1)],
            ["Total",   fmt_int(n),   fmt_num(100.0;   digits = 1)]]
    write_table(name; caption = caption, label = label,
                colspec = "p{3cm} p{3cm} p{3cm}",
                header = ["Decision", "Number", "Share (\\%)"],
                rows = rows, midrules = [2], notes = [note],
                n_agents = n, n_college = ncl, params...)
end

"""
    table_outcomes(models, labels, name; ...)

End-of-family outcomes across arms, in the format of `resource_summary.tex`.
`models` are parent models that have been simulated.
"""
function table_outcomes(models::Vector, labels::Vector{<:AbstractString}, name::AbstractString;
                        caption::AbstractString = "End-of-Family Outcomes",
                        label::AbstractString = "outcomes_summary",
                        rescale::Real = 10,
                        note::AbstractString = "Means computed over the simulated cohort at the end of the family stage.",
                        params...)
    @assert length(models) == length(labels)
    T = models[1].T
    fa = [mean(filter(isfinite, m.sim_a[:, T+1])) * rescale for m in models]
    fh = [mean(filter(isfinite, m.sim_hc[:, T+1]))          for m in models]
    rows = [vcat(["Mean final assets\\tnote{a}"], [fmt_num(v; digits = 2) for v in fa]),
            vcat(["Mean final human capital"],    [fmt_num(v; digits = 4) for v in fh])]
    write_table(name; caption = caption, label = label,
                colspec = "p{5.2cm} " * repeat("p{2.8cm} ", length(models)),
                header = vcat(["Outcome (at \$t=$(T+1)\$)"], collect(labels)),
                rows = rows,
                tnotes = ["a" => "Values in thousands of U.S.\\ dollars."],
                notes = [note], params...)
end

"""
    table_belief_groups(belief_type, belief_values, init_assets, final_assets, final_hc, name; ...)

Per-belief-group outcomes, in the format of `hetero_table.tex`.
"""
function table_belief_groups(belief_type, belief_values, init_assets, final_assets, final_hc,
                             name::AbstractString;
                             caption::AbstractString = "Outcomes by Subjective Belief Group (Assets \$\\times 10^3\$)",
                             label::AbstractString = "belief_groups",
                             note::AbstractString = "Means by belief group; belief value is the subjective annual human-capital increment from college.",
                             params...)
    rows = Vector{Vector{String}}()
    for m in 1:maximum(belief_type)
        idx = findall(==(m), belief_type)
        isempty(idx) && continue
        push!(rows, [fmt_int(m),
                     fmt_num(belief_values[m]; digits = 3),
                     fmt_num(mean(filter(isfinite, init_assets[idx]));  digits = 3),
                     fmt_num(mean(filter(isfinite, final_assets[idx])); digits = 3),
                     fmt_num(mean(filter(isfinite, final_hc[idx]));     digits = 3),
                     fmt_int(length(idx))])
    end
    write_table(name; caption = caption, label = label, colspec = "cccccc", small = true,
                header = ["Belief Group", "Belief Value", "Init.\\ Asset",
                          "Mean Final Asset", "Mean Final HC", "N Agents"],
                rows = rows, notes = [note], n_groups = length(rows), params...)
end

"""
    table_diagnostics(entries, name; ...)

Solver / accuracy diagnostics as a table, so the numbers behind "the solution is sound"
appear in the paper rather than only in a console log.
"""
function table_diagnostics(entries::Vector{<:Pair}, name::AbstractString;
                           caption::AbstractString = "Numerical Diagnostics",
                           label::AbstractString = "diagnostics",
                           note::AbstractString = "Diagnostics computed on the solved model and the simulated cohort. See docs/ERRORS.md for definitions.",
                           params...)
    rows = [[latex_escape(string(k)), string(v)] for (k, v) in entries]
    write_table(name; caption = caption, label = label,
                colspec = "p{8cm} p{4cm}",
                header = ["Diagnostic", "Value"], rows = rows, notes = [note], params...)
end

# -----------------------------------------------------------------------------
# PDF build
# -----------------------------------------------------------------------------

"""
    build_tables_pdf(; filename = "all_tables", engine = "pdflatex", clean = true)

Compile every `.tex` in `output/tables/` into one reviewable PDF at
`output/reports/<filename>.pdf`.

The wrapper is generated here rather than kept as a checked-in file, so the PDF always
reflects exactly the tables currently on disk — no stale `\\input` list. Tables are
included alphabetically, each on its own page, with the source filename shown above it.

Returns the PDF path, or `nothing` if no LaTeX engine is available.
"""
function build_tables_pdf(; filename::AbstractString = "all_tables",
                          engine::AbstractString = "pdflatex", clean::Bool = true)
    texdir = tabpath()
    files  = sort(filter(f -> endswith(f, ".tex"), readdir(texdir)))
    if isempty(files)
        @warn "No .tex files in $texdir — nothing to compile"
        return nothing
    end
    if Sys.which(engine) === nothing
        @warn "LaTeX engine '$engine' not found; tables written but no PDF built"
        return nothing
    end

    io = IOBuffer()
    println(io, raw"""
\documentclass[11pt]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{booktabs, threeparttable, float, longtable, amsmath, amssymb}
\usepackage[colorlinks=true, linkcolor=blue]{hyperref}
\setlength{\parindent}{0pt}
\begin{document}
\begin{center}
  {\Large\bfseries Structural Estimation --- Result Tables}\\[4pt]
""")
    println(io, "  \\texttt{", replace(git_sha(), "_" => "\\_"), "} \\quad ",
                Dates.format(Dates.now(), "yyyy-mm-dd HH:MM"), "\\\\[2pt]")
    println(io, "  ", length(files), " tables\\\\[6pt]")
    println(io, raw"\end{center}")
    println(io, raw"\tableofcontents\newpage")
    for f in files
        stem = replace(f[1:end-4], "_" => "\\_")
        println(io, "\\section{\\texttt{", stem, "}}")
        println(io, "\\input{", f[1:end-4], "}")
        println(io, "\\clearpage")
    end
    println(io, raw"\end{document}")

    wrapper = joinpath(texdir, filename * ".tex")
    write(wrapper, String(take!(io)))

    outdir = reportpath()
    ok = true
    cd(texdir) do
        for _ in 1:2                        # twice, for the table of contents
            p = run(pipeline(`$engine -interaction=nonstopmode -halt-on-error $(filename).tex`;
                             stdout = devnull, stderr = devnull), wait = false)
            wait(p)
            ok &= success(p)
        end
    end
    pdf_src = joinpath(texdir, filename * ".pdf")
    if !ok || !isfile(pdf_src)
        @warn "LaTeX compilation failed; see $(joinpath(texdir, filename * ".log"))"
        return nothing
    end
    pdf_dst = joinpath(outdir, filename * ".pdf")
    mv(pdf_src, pdf_dst; force = true)
    if clean
        for ext in (".aux", ".log", ".out", ".toc")
            f = joinpath(texdir, filename * ext); isfile(f) && rm(f)
        end
        rm(wrapper; force = true)
    end
    @info "Built tables PDF" pdf_dst n_tables=length(files)
    return pdf_dst
end

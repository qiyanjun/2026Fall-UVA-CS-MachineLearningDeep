# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Jekyll course site for UVA CS 4774/6316 Machine Learning (Fall 2026), served by GitHub Pages from
`main`: https://qiyanjun.github.io/2026Fall-UVA-CS-MachineLearningDeep/

## Build and check

```bash
bundle exec jekyll build -q      # full local build into _site/ (the faraday-retry warning is harmless)
bundle exec jekyll serve         # local preview at http://localhost:4000/2026Fall-UVA-CS-MachineLearningDeep/
gh run list -L 3                 # GitHub Pages deploy status after a push
```

There are no tests. After editing posts, rebuild and check the generated page in
`_site/contents/<file-stem>/index.html`. Useful checks: a `<table>` without `<thead>` is a paragraph
that kramdown turned into a table (see pipes below); literal `**` in the page text is broken bold.

## How the schedule is built

- One lecture = one file in `_contents/`. The schedule in `index.html`, `LecturesByDate.md` and
  `LecturesByTags.md` lists them **in file-name order**. GitHub Pages runs Jekyll 3.9, which ignores
  `sort_by: LOrder` in `_config.yml`, so `LOrder` is cosmetic: to move a lecture, rename its file
  (e.g. `S3-L23-Agents.md` → `S3-L24a-Agents.md` to sort after `S3-L24-Multimodal.md`). A rename changes
  the page URL (`/contents/<file-stem>/`), so grep for the old stem (announcements in `z26fBlog.md`
  link to posts).
- The "L12, L13, …" numbers on the schedule are a running counter over posts that are not `extra`,
  `background`, `platform` or `exam`; dates, quiz and HW columns come from the slot arrays near the top
  of `index.html`. These schedule numbers differ from the deck file numbers (deck `S3-L20-Decoder_Inference`
  is schedule row L21). Notebooks and quiz forms are named after the **deck** number.
- The "WE ARE HERE" row is placed before the first post whose `lectureVersion` is not `current`
  (currently `lectureVersion: next` in `S3-L20-DecoderInference.md`).
- A unified naming rule (post, deck and notebooks share one stem, e.g. `S3-L21-DecoderInference`) is
  planned for the next term. Until it is adopted, keep current names.
- Do not change the schedule table or slot arrays in `index.html` unless asked. Canvas is the source of
  truth for HW/quiz/exam details; `Assignments.md` keeps its reference tables.

## Post front matter (`_contents/*.md`)

Rendered by `_layouts/post.html` → `_includes/lecture-sections.html`. Do not edit those templates or
`LecturesByTags.md` to support new field shapes; use the existing fields:

- `lecture:` one slide PDF name in `Lectures/` without `.pdf`. A second PDF goes in `extraContent:`
  (shown as "Advanced Read").
- `notes:` / `morenotes:` readings, quoted string of `<a href=... target="_blank">` links separated by
  ` · ` (` + ` also accepted). Format: `Author et al. (Year), <a ...>Title</a>`.
- `notebooks:` coding resources, same separator. Repo notebooks link through Colab:
  `https://colab.research.google.com/github/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep/blob/main/notebook/<file>.ipynb`.
- `video:` modules as `<b>M1</b> <a>Original</a> · <a>AI-Clone</a> · <a>Author-Voice</a>  |  <b>M2</b> ...`
  (two spaces around `|`).
- Row type flags: `extra: true`, `background: true`, `platform: true`, `exam: true`; a title containing
  "Section" renders as a section header.

## Post body conventions

- The body is generated from the slide deck: `### In this lecture, we cover:` (nested `-`/`  +` list),
  an italic cross-reference line, then `# Study Guide: <title>` with a short-answer quiz, an answer key
  (each answer starts with a bolded sentence), essay questions, and a `| Term | Definition |` glossary.
  When a deck changes, update the post from the deck PDF; every claim must be supported by the deck.
- Refer to other lectures by **title**, never by lecture number.
- kramdown: a line containing `|` outside a real table becomes a one-row table and the pipe vanishes.
  Write `\|` (e.g. `P(y\|x)`, `\|\|x\|\|`). Write `<` as `&lt;` in math text.
- Front matter values with `'` inside single quotes must double it (`''`); a YAML error breaks the
  whole Pages build.

## Notebooks (`notebook/`)

- Newer teaching notebooks come in pairs, `Lnn-Topic-notebook-slug.ipynb` (student: `TODO` stubs raise
  `NotImplementedError`, the notebook still runs top to bottom) and `...-solutions.ipynb`. Publish a
  solutions file only after the homework it overlaps is past due.
- Notebook text must not name the course, semester or lecture numbers.
- An instructor archive keeps identical copies of every notebook here (plus unpublished solutions);
  copy changes there too (location in `CLAUDE.local.md`).
- Build or refactor notebooks with the `notebook-builder` skill (one source file → both versions,
  checked by executing them).

## Related places

Private locations (slide sources, instructor archive, homeworks, quiz forms, course plugins, the
planned-rename table) are listed in `CLAUDE.local.md`, which is not committed. This repo is public:
never write private paths or links to private repos in committed files.

## Commits

Updated slide PDFs in `Lectures/` are often modified by the instructor outside Claude; commit them
separately from post edits. Commit and push only when asked.

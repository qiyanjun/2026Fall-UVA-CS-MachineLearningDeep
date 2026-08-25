# 2026-Fall-UVA-CS-MachineLearningDeep

Course website (Jekyll site, served via GitHub Pages) for UVA CS's Fall 2026 Machine Learning / Deep Learning course, taught by Prof. Yanjun Qi.

**Live site:** https://qiyanjun.github.io/2026Fall-UVA-CS-MachineLearningDeep/

## File Structure

```
.
├── _config.yml            # Jekyll site config (title, baseurl, permalinks, collections)
├── Gemfile                # Ruby/Jekyll dependencies (github-pages gem)
├── index.html             # Home page: renders the lecture schedule table
├── About.md               # "Syllabus" nav page: course info, prerequisites, staff, policies
├── Assignments.md         # "Assignments" nav page: HWs, quizzes, grading breakdown
├── LecturesByDate.md      # "All-In-OnePage" nav page: every lecture's content in one long page
├── LecturesByTags.md      # "Topics-as-Tables" nav page: lectures grouped/indexed by topic tag
├── z26fBlog.md            # "Annoucements" nav page: dated course announcements/updates
├── 404.html                # GitHub Pages 404 error page
├── atom.xml                # Auto-generated RSS/Atom feed of lecture posts
├── LICENSE
│
├── _contents/              # One .md file per lecture (the "posts" collection) — see Module Structure below
├── _layouts/                # Jekyll page templates
│   ├── default.html         #   base HTML wrapper (loads sidebar/head, used by all pages)
│   ├── page.html             #   layout for standalone nav pages (About, Assignments, ...)
│   └── post.html             #   layout for a single lecture page under _contents/
├── _includes/               # Reusable HTML fragments
│   ├── head.html             #   <head> block (meta tags, CSS)
│   └── sidebar.html          #   left nav sidebar; auto-lists any page with `layout: page`
│
├── Lectures/                # Static assets: lecture PDFs (slides) linked from _contents/*.md
├── notebook/                 # Jupyter notebooks and quiz/review PDFs used as lecture resources
└── public/                   # Site assets: CSS, logo, favicon, images
```

Each file in `_contents/` is a Jekyll collection item with YAML front matter (`title`, `lecture`, `video`, `notes`, `categories`, `tags`, `lectureVersion`, ...) that `index.html`, `LecturesByDate.md`, and `LecturesByTags.md` read to build the schedule/index tables. `lecture: <name>` points to a PDF of the same name in `Lectures/`.

## Module Structure

Lectures are organized into 7 sections (`S0`–`S6`), reflected in the `_contents/` filename prefixes (e.g. `S2-L04-CNN.md`). Each section starts with a `S<N>-00Start.md` overview file.

| Section | Theme | # Lectures |
|---|---|---|
| S0 | Introduction & math prerequisites (algebra/calculus review) | 2 |
| S1 | Basics of Supervised Learning on Tabular Data (linear/regularized regression, kNN, model selection, bias-variance) | 11 |
| S2 | Deep Learning on 2D Grid Data / Imaging (MLE, logistic regression, NN, CNN, PyTorch/Keras/HuggingFace, PCA) | 9 |
| S3 | Deep Learning on 1D Sequence Data / Language (text NNs, generative & naive Bayes classification, recent DL/LLM survey) | 7 |
| S4 | More Advanced Supervised Learning on Tabular Data (SVM + kernels + duality, decision trees, bagging, boosting) | 8 |
| S5 | Unsupervised Learning (hierarchical & k-means clustering, GMM/EM, reinforcement learning) | 7 |
| S6 | Wrap-up (review, final exam, final project) | 4 |

## Important Links & Their Functions

| Link | Purpose |
|---|---|
| [Live course site](https://qiyanjun.github.io/2026Fall-UVA-CS-MachineLearningDeep/) | Main entry point — lecture-by-lecture schedule table with dates, slides, videos, and notes |
| [Syllabus](https://qiyanjun.github.io/2026Fall-UVA-CS-MachineLearningDeep/About/) (`About.md`) | Course description, learning goals, prerequisites, instructor/TA contacts & office hours, grading policy |
| [Assignments](https://qiyanjun.github.io/2026Fall-UVA-CS-MachineLearningDeep/Assignments/) (`Assignments.md`) | HW list with out/in dates and weights, quiz schedule, midterm/final exam info, grading breakdown |
| [All-In-OnePage](https://qiyanjun.github.io/2026Fall-UVA-CS-MachineLearningDeep/LecturesByDate/) (`LecturesByDate.md`) | Every lecture's full content rendered on a single scrollable page, in schedule order |
| [Topics-as-Tables](https://qiyanjun.github.io/2026Fall-UVA-CS-MachineLearningDeep/LecturesByTags/) (`LecturesByTags.md`) | Lectures indexed/grouped by topic tag, for topic-based lookup instead of date order |
| [Annoucements](https://qiyanjun.github.io/2026Fall-UVA-CS-MachineLearningDeep/z26fBlog/) (`z26fBlog.md`) | Dated log of course announcements and updates (also mirrored in Canvas) |
| [GitHub repo](https://github.com/qiyanjun/2026Fall-UVA-CS-MachineLearningDeep) | Source of this site; `Lectures/` (slide PDFs) and `notebook/` (code notebooks, quiz/review PDFs) are browsable directly here |
| [UVA Academic Calendar](http://www.virginia.edu/registrar/calendar.html) | Official university dates (drop deadlines, holidays, exam period) referenced on the home page |

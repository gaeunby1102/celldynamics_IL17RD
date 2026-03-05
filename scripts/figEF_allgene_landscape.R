## ============================================================
## figEF_allgene_landscape.R
## All-gene KO Landscape + Gene Program Validation Figures
## Version: v1.0.0
## ============================================================

rm(list = ls())

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggrepel)
  library(patchwork)
  library(scales)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(forcats)
})

version_tag <- "v1.0.0"

# ── Paths ─────────────────────────────────────────────────────
BASE     <- "/data2/Atlas_Normal/IL17RD_scdiffeq/results/trial6"
SCAN_DIR <- file.path(BASE, "allgene_scan")
VAL_DIR  <- file.path(BASE, "gene_expression_recon", "gene_program_validation")
OUT_DIR  <- file.path(BASE, "gene_expression_recon", "figures_R")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ── Theme ─────────────────────────────────────────────────────
theme_step1 <- function(base_size = 11) {
  theme(
    title        = element_text(size = 14, color = "black"),
    text         = element_text(size = 12, color = "black"),
    axis.title   = element_text(size = 14, color = "black"),
    axis.text    = element_text(size = 12, color = "black"),
    strip.text   = element_text(size = 12, color = "black"),
    strip.background = element_rect(fill = "white", color = "black"),
    panel.grid.major = element_line(color = alpha("lightgrey", 0.5)),
    panel.grid.minor = element_line(color = alpha("lightgrey", 0.3)),
    panel.border     = element_blank(),
    panel.background = element_rect(fill = "white", colour = NA),
    axis.line    = element_line(colour = "black", linewidth = 0.6),
    legend.background = element_rect(color = NA, fill = "white"),
    legend.key   = element_rect(fill = "white"),
    legend.title = element_text(size = 12),
    legend.text  = element_text(size = 10),
    plot.margin  = unit(c(0.4, 0.8, 0.4, 0.4), "cm"),
    plot.tag     = element_text(size = 16, face = "bold", color = "black")
  )
}

# ── Colors ────────────────────────────────────────────────────
gene_colors <- c(
  IL17RD  = "#E74C3C",
  PAX6    = "#E67E22",
  NEUROG2 = "#F39C12",
  ASCL1   = "#27AE60",
  DLX2    = "#8E44AD",
  HES1    = "#795548"
)
REF_GENES <- names(gene_colors)

# ── Load data ─────────────────────────────────────────────────
bio70  <- read_csv(file.path(SCAN_DIR, "allgene_ko_scan_t70d_RG_bio.csv"),  show_col_types = FALSE)
bio115 <- read_csv(file.path(SCAN_DIR, "allgene_ko_scan_t115d_RG_bio.csv"), show_col_types = FALSE)
sim70  <- read_csv(file.path(SCAN_DIR, "topgene_sim_t70d_RG.csv"),  show_col_types = FALSE)
sim115 <- read_csv(file.path(SCAN_DIR, "topgene_sim_t115d_RG.csv"), show_col_types = FALSE)

# Label ref genes
bio70  <- bio70  %>% mutate(is_ref = gene %in% REF_GENES)
bio115 <- bio115 %>% mutate(is_ref = gene %in% REF_GENES)

# ============================================================
# Fig E — All-gene KO Landscape
# E1: Landscape scatter (bio_rank vs ko_l2_mean)
# E2: Scan vs Simulation scatter
# E3: Top 20 barplot
# ============================================================

# ── E1: landscape scatter helper ─────────────────────────────
make_landscape <- function(bio_df, timepoint_label, tag) {
  ref_df  <- bio_df %>% filter(is_ref)
  rest_df <- bio_df %>% filter(!is_ref)

  ggplot() +
    geom_point(data = rest_df,
               aes(x = bio_rank, y = ko_l2_mean),
               color = "#AAAAAA", size = 0.8, alpha = 0.5) +
    geom_point(data = ref_df,
               aes(x = bio_rank, y = ko_l2_mean, color = gene),
               size = 3.5, alpha = 0.9) +
    geom_text_repel(data = ref_df,
                    aes(x = bio_rank, y = ko_l2_mean, label = gene, color = gene),
                    size = 3.5, fontface = "italic",
                    box.padding = 0.4, point.padding = 0.3,
                    min.segment.length = 0.2, max.overlaps = 20,
                    show.legend = FALSE) +
    scale_color_manual(values = gene_colors, name = NULL) +
    labs(
      title = sprintf("All-gene KO landscape  [%s]", timepoint_label),
      x     = "Biological rank (by KO L2, HK-filtered)",
      y     = "Mean paired L2 at t=0",
      tag   = tag
    ) +
    theme_step1() +
    theme(legend.position = "right")
}

pE1a <- make_landscape(bio70,  "t70d RG",  "E1")
pE1b <- make_landscape(bio115, "t115d RG", "E2")

# ── E2: Scan vs Simulation scatter ───────────────────────────
make_scan_sim <- function(sim_df, timepoint_label, tag) {
  r_val <- cor(sim_df$scan_l2, sim_df$sim_l2_end, method = "spearman")

  ref_df  <- sim_df %>% filter(is_ref)
  rest_df <- sim_df %>% filter(!is_ref)

  ggplot() +
    geom_point(data = rest_df,
               aes(x = scan_l2, y = sim_l2_end),
               color = "#555555", size = 1.5, alpha = 0.4) +
    geom_point(data = ref_df,
               aes(x = scan_l2, y = sim_l2_end, color = gene),
               size = 4, alpha = 0.95) +
    geom_text_repel(data = ref_df,
                    aes(x = scan_l2, y = sim_l2_end, label = gene, color = gene),
                    size = 3.5, fontface = "italic",
                    box.padding = 0.4, point.padding = 0.3,
                    min.segment.length = 0.2, max.overlaps = 20,
                    show.legend = FALSE) +
    scale_color_manual(values = gene_colors, name = NULL) +
    labs(
      title = sprintf("Scan vs Simulation  [%s]\nSpearman r = %.3f", timepoint_label, r_val),
      x     = "KO L2 at t=0 (encoding scan)",
      y     = "KO L2 at t=1.0 (trajectory simulation)",
      tag   = tag
    ) +
    theme_step1() +
    theme(legend.position = "right")
}

pE2a <- make_scan_sim(sim70  %>% mutate(is_ref = gene %in% REF_GENES), "t70d RG",  "E3")
pE2b <- make_scan_sim(sim115 %>% mutate(is_ref = gene %in% REF_GENES), "t115d RG", "E4")

# ── E3: Top 20 barplot ────────────────────────────────────────
make_top20_bar <- function(sim_df, timepoint_label, tag) {
  top20 <- sim_df %>%
    arrange(desc(sim_l2_end)) %>%
    slice_head(n = 20) %>%
    mutate(
      gene  = fct_reorder(gene, sim_l2_end),
      color = ifelse(gene %in% REF_GENES, as.character(gene_colors[as.character(gene)]), "#AAAAAA")
    )

  ggplot(top20, aes(x = gene, y = sim_l2_end, fill = gene)) +
    geom_col(width = 0.7, show.legend = FALSE) +
    scale_fill_manual(
      values = setNames(
        ifelse(levels(top20$gene) %in% REF_GENES,
               gene_colors[levels(top20$gene)],
               "#AAAAAA"),
        levels(top20$gene)
      )
    ) +
    coord_flip() +
    labs(
      title = sprintf("Top 20 trajectory divergence  [%s]", timepoint_label),
      x     = NULL,
      y     = "KO L2 at t=1.0",
      tag   = tag
    ) +
    theme_step1() +
    theme(axis.text.y = element_text(size = 10, face = "italic"))
}

pE3a <- make_top20_bar(sim70,  "t70d RG",  "E5")
pE3b <- make_top20_bar(sim115, "t115d RG", "E6")

# ── Assemble Fig E ────────────────────────────────────────────
figE <- (pE1a | pE1b) / (pE2a | pE2b) / (pE3a | pE3b) +
  plot_annotation(
    title = "Figure E: All-gene KO Scan & Trajectory Landscape",
    theme = theme(plot.title = element_text(size = 16, face = "bold"))
  )

out_E <- file.path(OUT_DIR, sprintf("figE_allgene_landscape_%s", version_tag))
ggsave(paste0(out_E, ".pdf"), figE, width = 16, height = 18)
ggsave(paste0(out_E, ".png"), figE, width = 16, height = 18, dpi = 200)
cat(sprintf("✓ Saved: figE_allgene_landscape_%s\n", version_tag))


# ============================================================
# Fig F — Gene Program Validation
# Top 5 correlated genes per ref gene × 2 timepoints
# ============================================================

val_df <- read_csv(file.path(VAL_DIR, "gene_program_validation_summary.csv"),
                   show_col_types = FALSE)

# Known targets from literature (used for annotation)
known_targets <- list(
  IL17RD  = list(up = c("FGFR3", "NR2F1"),         down = c("AURKA", "CDC20")),
  PAX6    = list(up = c("EOMES", "TBR2", "SOX2", "LHX2", "EMX2"),
                 down = c("TBR1", "SATB2", "BCL11B")),
  HES1    = list(up = c("SFRP1"),                   down = c("VIM", "GFAP")),
  ASCL1   = list(up = c("DLX2", "NEUROD1"),         down = c("VIM")),
  NEUROG2 = list(up = c("EOMES", "TBR1", "MEIS2"),  down = c()),
  DLX2    = list(up = c("FOXP2", "EPHA6"),           down = c())
)

make_validation_panel <- function(df_val, ref_gene, timepoint_label, tag = NULL) {
  df_g <- df_val %>%
    filter(perturb_gene == ref_gene, t0_tag == timepoint_label) %>%
    arrange(desc(pearson_r)) %>%
    slice_head(n = 10)

  if (nrow(df_g) == 0) return(NULL)

  kt_all <- c(known_targets[[ref_gene]]$up, known_targets[[ref_gene]]$down)

  df_g <- df_g %>%
    mutate(
      top_gene    = fct_reorder(top_gene, pearson_r),
      is_known    = top_gene %in% kt_all,
      bar_color   = ifelse(is_known, gene_colors[ref_gene], "#BBBBBB"),
      label_face  = ifelse(is_known, "bold.italic", "italic")
    )

  p <- ggplot(df_g, aes(x = top_gene, y = pearson_r, fill = I(bar_color))) +
    geom_col(width = 0.7, show.legend = FALSE) +
    geom_hline(yintercept = 0, linewidth = 0.4, color = "black") +
    coord_flip() +
    labs(
      title = sprintf("%s  [%s]",
                      sub("_RG", "", timepoint_label) %>% sub("t", "t=", .),
                      ref_gene),
      x     = NULL,
      y     = "Pearson r (KO direction × gene expr)"
    ) +
    theme_step1(base_size = 10) +
    theme(
      plot.title  = element_text(size = 12, face = "bold",
                                 color = gene_colors[ref_gene]),
      axis.text.y = element_text(
        size  = 9,
        face  = df_g %>% arrange(pearson_r) %>% pull(label_face),
        color = ifelse(df_g %>% arrange(pearson_r) %>% pull(is_known),
                       gene_colors[ref_gene], "black")
      ),
      plot.margin = unit(c(0.2, 0.5, 0.2, 0.2), "cm")
    )
  if (!is.null(tag)) p <- p + labs(tag = tag)
  p
}

# Build 6×2 grid (gene × timepoint)
panel_list <- list()
tag_idx <- 1
tag_letters <- LETTERS
for (gene in REF_GENES) {
  for (tp in c("t70d_RG", "t115d_RG")) {
    tag_lbl <- paste0("F", tag_idx)
    panel_list[[length(panel_list) + 1]] <-
      make_validation_panel(val_df, gene, tp, tag = tag_lbl)
    tag_idx <- tag_idx + 1
  }
}
panel_list <- Filter(Negate(is.null), panel_list)

# Assemble in 4-column layout (pairs: t70 | t115 per gene, 2 genes per row)
n_panels <- length(panel_list)
figF <- wrap_plots(panel_list, ncol = 4) +
  plot_annotation(
    title   = "Figure F: Gene Program Validation - Top Correlated Genes",
    caption = "Highlighted bars = known targets from literature",
    theme   = theme(
      plot.title   = element_text(size = 16, face = "bold"),
      plot.caption = element_text(size = 10, color = "grey40")
    )
  )

out_F <- file.path(OUT_DIR, sprintf("figF_geneprogram_validation_%s", version_tag))
ggsave(paste0(out_F, ".pdf"), figF, width = 18, height = 14)
ggsave(paste0(out_F, ".png"), figF, width = 18, height = 14, dpi = 200)
cat(sprintf("✓ Saved: figF_geneprogram_validation_%s\n", version_tag))


# ============================================================
# Fig G (optional) — IL17RD focus: KO effect comparison
# IL17RD rank in landscape + known targets highlighted
# ============================================================

make_il17rd_focus <- function(bio_df, sim_df, tp_label) {
  il17rd_bio <- bio_df %>% filter(gene == "IL17RD")
  il17rd_sim <- sim_df %>% filter(gene == "IL17RD")

  # Sub-panel 1: rank context in bio landscape
  n_total <- nrow(bio_df)
  p1 <- ggplot(bio_df, aes(x = bio_rank, y = ko_l2_mean)) +
    geom_point(color = "#CCCCCC", size = 0.6, alpha = 0.5) +
    geom_point(data = il17rd_bio,
               aes(x = bio_rank, y = ko_l2_mean),
               color = gene_colors["IL17RD"], size = 4) +
    geom_text_repel(data = il17rd_bio,
                    aes(x = bio_rank, y = ko_l2_mean, label = sprintf("IL17RD\n(rank %d/%d)", bio_rank, n_total)),
                    color = gene_colors["IL17RD"], size = 3.5, fontface = "bold.italic",
                    nudge_x = n_total * 0.1, nudge_y = 0.01) +
    labs(title = sprintf("IL17RD KO rank  [%s]", tp_label),
         x = "Biological rank", y = "KO L2 at t=0") +
    theme_step1()

  # Sub-panel 2: scan vs sim for all ref genes, sized by sim rank
  ref_sim <- sim_df %>% filter(gene %in% REF_GENES) %>%
    mutate(gene = factor(gene, levels = REF_GENES))

  p2 <- ggplot(ref_sim, aes(x = scan_l2, y = sim_l2_end, color = gene, label = gene)) +
    geom_point(size = 4) +
    geom_text_repel(size = 3.5, fontface = "italic", box.padding = 0.5,
                    show.legend = FALSE) +
    scale_color_manual(values = gene_colors, name = NULL) +
    labs(title = sprintf("Ref gene KO effects  [%s]", tp_label),
         x = "KO L2 at t=0 (scan)", y = "KO L2 at t=1.0 (simulation)") +
    theme_step1() +
    theme(legend.position = "none")

  p1 | p2
}

pG_70  <- make_il17rd_focus(bio70,  sim70,  "t70d RG")
pG_115 <- make_il17rd_focus(bio115, sim115, "t115d RG")

figG <- (pG_70 / pG_115) +
  plot_annotation(
    title = "Figure G: IL17RD in All-gene KO Context",
    theme = theme(plot.title = element_text(size = 16, face = "bold"))
  )

out_G <- file.path(OUT_DIR, sprintf("figG_IL17RD_focus_%s", version_tag))
ggsave(paste0(out_G, ".pdf"), figG, width = 14, height = 10)
ggsave(paste0(out_G, ".png"), figG, width = 14, height = 10, dpi = 200)
cat(sprintf("✓ Saved: figG_IL17RD_focus_%s\n", version_tag))

cat("\n=== All figures saved ===\n")
cat(sprintf("  Output directory: %s\n", OUT_DIR))

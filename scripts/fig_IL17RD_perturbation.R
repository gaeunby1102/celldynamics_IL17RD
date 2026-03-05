## ============================================================
## fig_IL17RD_perturbation.R
## IL17RD scDiffeq Perturbation Analysis Figures
## Version: v1.0.0
## ============================================================

rm(list = ls())

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggrepel)
  library(patchwork)
  library(cowplot)
  library(scales)
  library(dplyr)
  library(tidyr)
  library(readr)
})

version_tag <- "v1.0.0"

# ── Paths ─────────────────────────────────────────────────────
BASE    <- "/data2/Atlas_Normal/IL17RD_scdiffeq/results/trial6/gene_expression_recon"
OUT_DIR <- file.path(BASE, "figures_R")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ── Theme (user style) ────────────────────────────────────────
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

# color palettes
COL_POS     <- "#C0392B"   # positive correlation (KO direction)
COL_NEG     <- "#2980B9"   # negative correlation
COL_IL17RD  <- "#E74C3C"
COL_PAX6    <- "#E67E22"
COL_NEUROG2 <- "#F1C40F"
COL_ASCL1   <- "#27AE60"
COL_DLX2    <- "#8E44AD"
COL_HES1    <- "#795548"
COL_CTRL    <- "#95A5A6"

gene_colors <- c(
  IL17RD = COL_IL17RD, PAX6 = COL_PAX6,
  NEUROG2 = COL_NEUROG2, ASCL1 = COL_ASCL1,
  DLX2 = COL_DLX2, HES1 = COL_HES1
)

# ============================================================
# Fig A: Gene Program Correlation Barplot (t70d + t115d)
# ============================================================

df70  <- read_csv(file.path(BASE, "gene_program/gene_program_t70d_RG.csv"),
                  show_col_types = FALSE)
df115 <- read_csv(file.path(BASE, "gene_program/gene_program_t115d_RG.csv"),
                  show_col_types = FALSE)

make_gene_program_plot <- function(df, timepoint, n_top = 20,
                                   color_pos = COL_POS, color_neg = COL_NEG) {
  top <- bind_rows(
    df %>% arrange(desc(pearson_r)) %>% slice_head(n = n_top %/% 2),
    df %>% arrange(pearson_r)       %>% slice_head(n = n_top %/% 2)
  ) %>%
    distinct(gene, .keep_all = TRUE) %>%
    arrange(pearson_r) %>%
    mutate(
      gene   = factor(gene, levels = gene),
      dir    = ifelse(pearson_r > 0, "positive", "negative"),
      sig    = padj < 0.05,
      label  = gene
    )

  p <- ggplot(top, aes(x = pearson_r, y = gene, fill = dir)) +
    geom_col(aes(alpha = sig), width = 0.75) +
    geom_vline(xintercept = 0, color = "black", linewidth = 0.5) +
    geom_text(aes(label = label,
                  x = ifelse(pearson_r > 0, -0.005, 0.005),
                  hjust = ifelse(pearson_r > 0, 1, 0)),
              size = 3.3, color = "black") +
    scale_fill_manual(
      values = c(positive = color_pos, negative = color_neg),
      labels = c(positive = "Up (KO direction)", negative = "Down (KO direction)"),
      name   = ""
    ) +
    scale_alpha_manual(values = c("TRUE" = 1.0, "FALSE" = 0.4),
                       guide = "none") +
    scale_x_continuous(
      limits = c(-0.55, 0.55),
      breaks = seq(-0.4, 0.4, 0.2),
      labels = function(x) sprintf("%+.1f", x)
    ) +
    labs(
      title    = timepoint,
      subtitle = sprintf("IL17RD KO direction (PC1 of Δz, padj<0.05 opaque)"),
      x        = "Pearson r  (gene expr vs KO-direction score)",
      y        = NULL
    ) +
    theme_step1() +
    theme(
      axis.text.y  = element_blank(),
      axis.ticks.y = element_blank(),
      legend.position = "bottom"
    )
  return(p)
}

p_70  <- make_gene_program_plot(df70,  "t70d → t168d  (RG start)")
p_115 <- make_gene_program_plot(df115, "t115d → t168d  (RG start)")

fig_A <- (p_70 | p_115) +
  plot_annotation(
    title   = "Fig A. IL17RD KO-associated Gene Program in scVI Latent Space",
    subtitle = sprintf("Pearson r: gene expression ~ perturbation score  [%s]", version_tag),
    tag_levels = "a"
  )

ggsave(file.path(OUT_DIR, sprintf("figA_gene_program_%s.pdf", version_tag)),
       fig_A, width = 13, height = 7)
ggsave(file.path(OUT_DIR, sprintf("figA_gene_program_%s.png", version_tag)),
       fig_A, width = 13, height = 7, dpi = 200)
cat("  Fig A saved.\n")

# ============================================================
# Fig B: Uncertainty Quantification — L2 with 95% CI
# ============================================================

unc <- read_csv(file.path(BASE, "uncertainty/perturbation_uncertainty.csv"),
                show_col_types = FALSE)

# condition 파싱
unc <- unc %>%
  mutate(
    perturb_label = condition,
    gene_f = factor(gene, levels = c("IL17RD","PAX6","NEUROG2","ASCL1","DLX2","HES1")),
    type_f = factor(perturb_type, levels = c("KO","OE3x")),
    # 유전자별 색상
    gene_col = gene_colors[gene],
    # KO: solid, OE: lighter
    alpha_v  = ifelse(perturb_type == "KO", 1.0, 0.55)
  )

make_unc_plot <- function(df_t, t0_tag) {
  ctrl_noise_mean <- df_t$ctrl_noise_l2_mean[1]
  ctrl_noise_95th <- df_t$ctrl_noise_l2_95th[1]

  df_t <- df_t %>%
    arrange(desc(l2_mean)) %>%
    mutate(condition = factor(condition, levels = condition))

  p <- ggplot(df_t, aes(y = condition, x = l2_mean,
                         xmin = l2_lo, xmax = l2_hi,
                         color = gene, alpha = type_f)) +
    geom_vline(xintercept = ctrl_noise_mean, color = "grey50",
               linetype = "dashed", linewidth = 0.8) +
    geom_vline(xintercept = ctrl_noise_95th, color = "grey20",
               linetype = "dotted", linewidth = 0.8) +
    geom_errorbarh(height = 0.3, linewidth = 0.8) +
    geom_point(aes(shape = type_f), size = 3) +
    scale_color_manual(values = gene_colors, name = "Gene") +
    scale_alpha_manual(values = c(KO = 1.0, OE3x = 0.55), name = "Type") +
    scale_shape_manual(values  = c(KO = 16, OE3x = 17), name = "Type") +
    annotate("text", x = ctrl_noise_mean, y = Inf,
             label = sprintf("Ctrl noise\n(mean=%.3f)", ctrl_noise_mean),
             hjust = 1.1, vjust = 1.3, size = 3, color = "grey50") +
    annotate("text", x = ctrl_noise_95th, y = Inf,
             label = sprintf("95th\n(%.3f)", ctrl_noise_95th),
             hjust = -0.1, vjust = 1.3, size = 3, color = "grey20") +
    labs(
      title    = t0_tag,
      subtitle = sprintf("N=%d SDE runs · 95%% CI shown", df_t$n_runs[1]),
      x        = "Paired L2  (ctrl vs perturbation endpoint)",
      y        = NULL
    ) +
    theme_step1() +
    theme(legend.position = "right")
  return(p)
}

p_unc_70  <- make_unc_plot(unc %>% filter(t0_tag == "t70d_RG"),  "t70d RG start")
p_unc_115 <- make_unc_plot(unc %>% filter(t0_tag == "t115d_RG"), "t115d RG start")

fig_B <- (p_unc_70 / p_unc_115) +
  plot_annotation(
    title    = "Fig B. Perturbation Trajectory Divergence with Uncertainty (95% CI)",
    subtitle = sprintf("Dashed: ctrl noise mean · Dotted: ctrl noise 95th percentile  [%s]", version_tag),
    tag_levels = "a"
  )

ggsave(file.path(OUT_DIR, sprintf("figB_uncertainty_%s.pdf", version_tag)),
       fig_B, width = 10, height = 10)
ggsave(file.path(OUT_DIR, sprintf("figB_uncertainty_%s.png", version_tag)),
       fig_B, width = 10, height = 10, dpi = 200)
cat("  Fig B saved.\n")

# ============================================================
# Fig C: Gene Program Concordance (t70d vs t115d)
# ============================================================

conc <- read_csv(file.path(BASE, "gene_program/gene_program_concordance.csv"),
                 show_col_types = FALSE)

# 주요 유전자 하이라이트
highlight_genes <- c("IL17RD", "NR2F1", "FGFR3", "NFIA",
                     "AURKA", "CDC20", "CCNB1", "VIM",
                     "SPARCL1", "EDNRB", "ANK2", "ROBO2",
                     "HES1", "ASCL1", "NEUROG2")
conc <- conc %>%
  mutate(
    highlight = gene %in% highlight_genes,
    label     = ifelse(highlight, gene, NA_character_),
    pt_size   = ifelse(highlight, 2.5, 0.8),
    pt_alpha  = ifelse(highlight, 1.0, 0.25),
    quad      = case_when(
      pearson_r_70d > 0 & pearson_r_115d > 0 ~ "Up/Up",
      pearson_r_70d < 0 & pearson_r_115d < 0 ~ "Down/Down",
      TRUE ~ "Discordant"
    )
  )

cor_val <- cor(conc$pearson_r_70d, conc$pearson_r_115d, method = "spearman")

fig_C <- ggplot(conc, aes(x = pearson_r_70d, y = pearson_r_115d)) +
  geom_hline(yintercept = 0, color = "grey70", linewidth = 0.4) +
  geom_vline(xintercept = 0, color = "grey70", linewidth = 0.4) +
  geom_point(data = conc %>% filter(!highlight),
             size = 0.8, alpha = 0.2, color = "grey60") +
  geom_point(data = conc %>% filter(highlight),
             aes(color = quad), size = 2.8, alpha = 0.9) +
  geom_text_repel(data = conc %>% filter(highlight),
                  aes(label = label, color = quad),
                  size = 3.2, max.overlaps = 20,
                  box.padding = 0.4, segment.color = "grey60",
                  segment.size = 0.3) +
  scale_color_manual(
    values = c("Up/Up" = COL_POS, "Down/Down" = COL_NEG, "Discordant" = "grey40"),
    name   = "Direction"
  ) +
  annotate("text", x = -Inf, y = Inf,
           label = sprintf("Spearman r = %.3f", cor_val),
           hjust = -0.1, vjust = 1.3, size = 4.5, color = "black",
           fontface = "bold") +
  labs(
    title    = "Fig C. Gene Program Concordance: t70d vs t115d RG",
    subtitle = sprintf("IL17RD KO-direction gene correlations  [%s]", version_tag),
    x        = "Pearson r (t70d RG)",
    y        = "Pearson r (t115d RG)",
    tag      = "c"
  ) +
  theme_step1()

ggsave(file.path(OUT_DIR, sprintf("figC_concordance_%s.pdf", version_tag)),
       fig_C, width = 7, height = 6)
ggsave(file.path(OUT_DIR, sprintf("figC_concordance_%s.png", version_tag)),
       fig_C, width = 7, height = 6, dpi = 200)
cat("  Fig C saved.\n")

# ============================================================
# Fig D: IL17RD Ranking Summary (3-way model concordance)
# ============================================================

# 3-way ranking을 divergence CSV에서 복원
make_ranking_data <- function() {
  tags <- c("enforce1", "trial7", "trial8")
  dfs  <- list()
  for (tg in tags) {
    f <- sprintf(
      "/data2/Atlas_Normal/IL17RD_scdiffeq/results/trial6/traj_divergence/divergence_l2_%s_t70d_RG.csv",
      tg
    )
    if (!file.exists(f)) next
    d <- read_csv(f, show_col_types = FALSE) %>%
      slice_tail(n = 1) %>%
      pivot_longer(-t, names_to = "condition", values_to = "l2") %>%
      mutate(model = tg, t0_tag = "t70d_RG")
    dfs[[tg]] <- d
  }
  for (tg in tags) {
    f <- sprintf(
      "/data2/Atlas_Normal/IL17RD_scdiffeq/results/trial6/traj_divergence/divergence_l2_%s_t115d_RG.csv",
      tg
    )
    if (!file.exists(f)) next
    d <- read_csv(f, show_col_types = FALSE) %>%
      slice_tail(n = 1) %>%
      pivot_longer(-t, names_to = "condition", values_to = "l2") %>%
      mutate(model = tg, t0_tag = "t115d_RG")
    dfs[[paste0(tg, "_115")]] <- d
  }
  bind_rows(dfs)
}

rank_df <- make_ranking_data()

if (nrow(rank_df) > 0) {
  rank_df <- rank_df %>%
    group_by(model, t0_tag) %>%
    mutate(rank = rank(-l2)) %>%
    ungroup() %>%
    mutate(
      gene = gsub("_(KO|OE)$", "", condition),
      type = ifelse(grepl("KO$", condition), "KO", "OE"),
      model_label = recode(model,
        enforce1 = "Trial5\n(enforce1)",
        trial7   = "Trial7\n(116d holdout)",
        trial8   = "Trial8\n(tuned)"
      ),
      is_il17rd = gene == "IL17RD"
    )

  fig_D <- ggplot(rank_df %>% filter(type == "KO"),
                  aes(x = model_label, y = l2,
                      group = condition, color = is_il17rd)) +
    geom_line(aes(alpha = is_il17rd), linewidth = 0.8) +
    geom_point(aes(size = is_il17rd, shape = is_il17rd), alpha = 0.9) +
    geom_text_repel(
      data = rank_df %>% filter(type == "KO", model == "enforce1"),
      aes(label = condition), size = 2.8, nudge_x = -0.3,
      segment.size = 0.3, max.overlaps = 15
    ) +
    scale_color_manual(values = c("TRUE" = COL_IL17RD, "FALSE" = "grey70"),
                       guide = "none") +
    scale_alpha_manual(values = c("TRUE" = 1.0, "FALSE" = 0.4), guide = "none") +
    scale_size_manual(values  = c("TRUE" = 3.5, "FALSE" = 1.8), guide = "none") +
    scale_shape_manual(values = c("TRUE" = 16,  "FALSE" = 16),  guide = "none") +
    facet_wrap(~ t0_tag, nrow = 1,
               labeller = as_labeller(c(t70d_RG  = "t70d RG → t168d",
                                        t115d_RG = "t115d RG → t168d"))) +
    labs(
      title    = "Fig D. 3-Model Perturbation Concordance (KO, L2 at t=1.0)",
      subtitle = sprintf("Red = IL17RD_KO  ·  Grey = other genes  [%s]", version_tag),
      x        = NULL, y = "Paired L2  (t=1.0)",
      tag      = "d"
    ) +
    theme_step1() +
    theme(strip.background = element_rect(fill = "grey95", color = "grey70"))

  ggsave(file.path(OUT_DIR, sprintf("figD_ranking_%s.pdf", version_tag)),
         fig_D, width = 11, height = 6)
  ggsave(file.path(OUT_DIR, sprintf("figD_ranking_%s.png", version_tag)),
         fig_D, width = 11, height = 6, dpi = 200)
  cat("  Fig D saved.\n")
}

# ============================================================
# Combined panel figure
# ============================================================
cat("\nAll figures saved to:", OUT_DIR, "\n")
cat("Files:\n")
cat(paste0("  ", list.files(OUT_DIR), collapse = "\n"), "\n")

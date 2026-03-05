## ============================================================
## residual_l2_ranking.R
## Expression-corrected KO L2 ranking (Residual Analysis)
## Removes confounding of high expression on L2 score
## Version: v1.0.0
## ============================================================

rm(list = ls())

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggrepel)
  library(patchwork)
  library(scales)
  library(dplyr)
  library(readr)
})

version_tag <- "v1.0.0"

BASE     <- "/data2/Atlas_Normal/IL17RD_scdiffeq/results/trial6"
SCAN_DIR <- file.path(BASE, "allgene_scan")
OUT_DIR  <- file.path(BASE, "gene_expression_recon", "figures_R")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

theme_step1 <- function() {
  theme(
    text         = element_text(size = 12, color = "black"),
    axis.title   = element_text(size = 13, color = "black"),
    axis.text    = element_text(size = 11, color = "black"),
    panel.grid.major = element_line(color = scales::alpha("lightgrey", 0.5)),
    panel.grid.minor = element_line(color = scales::alpha("lightgrey", 0.3)),
    panel.border     = element_blank(),
    panel.background = element_rect(fill = "white", colour = NA),
    axis.line    = element_line(colour = "black", linewidth = 0.6),
    legend.background = element_rect(color = NA, fill = "white"),
    legend.key   = element_rect(fill = "white"),
    plot.margin  = unit(c(0.4, 0.8, 0.4, 0.4), "cm"),
    plot.tag     = element_text(size = 15, face = "bold")
  )
}

gene_colors <- c(
  IL17RD  = "#E74C3C", PAX6    = "#E67E22",
  NEUROG2 = "#F39C12", ASCL1   = "#27AE60",
  DLX2    = "#8E44AD", HES1    = "#795548"
)
REF_GENES <- names(gene_colors)

# ── Load & fit regression per timepoint ──────────────────────
process_timepoint <- function(tp_tag, tp_label) {
  df <- read_csv(file.path(SCAN_DIR, paste0("allgene_ko_scan_", tp_tag, "_bio.csv")),
                 show_col_types = FALSE) %>%
    mutate(
      log_mean_expr = log1p(mean_expr),
      is_ref        = gene %in% REF_GENES
    )

  # Fit regression: ko_l2_mean ~ log(mean_expr) + frac_expr
  fit <- lm(ko_l2_mean ~ log_mean_expr + frac_expr, data = df)
  df$residual_l2 <- residuals(fit)
  df$resid_rank  <- rank(-df$residual_l2, ties.method = "first")

  cat(sprintf("\n[%s] Regression R^2 = %.4f\n", tp_label, summary(fit)$r.squared))
  cat(sprintf("  (%.1f%% of L2 variance explained by expression level)\n",
              summary(fit)$r.squared * 100))

  list(df = df, fit = fit, tp_label = tp_label, tp_tag = tp_tag)
}

res70  <- process_timepoint("t70d_RG",  "t70d RG")
res115 <- process_timepoint("t115d_RG", "t115d RG")

# Save corrected rankings
for (res in list(res70, res115)) {
  out_df <- res$df %>%
    select(gene, bio_rank, ko_l2_mean, residual_l2, resid_rank,
           mean_expr, frac_expr, is_ref) %>%
    arrange(resid_rank)
  write_csv(out_df, file.path(SCAN_DIR,
    paste0("allgene_ko_scan_", res$tp_tag, "_residual.csv")))
  cat(sprintf("Saved: allgene_ko_scan_%s_residual.csv\n", res$tp_tag))
}

# ── Print top 30 residual-ranked genes ───────────────────────
for (res in list(res70, res115)) {
  cat(sprintf("\n--- Top 30 residual-corrected genes [%s] ---\n", res$tp_label))
  top30 <- res$df %>% arrange(resid_rank) %>%
    slice_head(n = 30) %>%
    select(resid_rank, gene, ko_l2_mean, residual_l2, bio_rank, mean_expr)
  print(as.data.frame(top30), row.names = FALSE)
}

# ── Ref gene ranks: raw vs corrected ─────────────────────────
cat("\n--- Reference gene: raw rank vs residual rank ---\n")
cat(sprintf("%-12s  %10s  %10s  %10s  %10s\n",
            "Gene", "t70_raw", "t70_resid", "t115_raw", "t115_resid"))
for (g in REF_GENES) {
  r70  <- res70$df  %>% filter(gene == g)
  r115 <- res115$df %>% filter(gene == g)
  cat(sprintf("%-12s  %10d  %10d  %10d  %10d\n",
              g,
              if (nrow(r70 ) > 0) r70$bio_rank[1]   else NA,
              if (nrow(r70 ) > 0) r70$resid_rank[1]  else NA,
              if (nrow(r115) > 0) r115$bio_rank[1]  else NA,
              if (nrow(r115) > 0) r115$resid_rank[1] else NA))
}

# ============================================================
# Figures
# ============================================================

make_panels <- function(res) {
  df       <- res$df
  tp_label <- res$tp_label
  ref_df   <- df %>% filter(is_ref)
  other_df <- df %>% filter(!is_ref)
  n_total  <- nrow(df)

  # Panel 1: raw L2 vs expression (show confound)
  p1 <- ggplot(df, aes(x = log_mean_expr, y = ko_l2_mean)) +
    geom_point(data = other_df, color = "#BBBBBB", size = 0.7, alpha = 0.4) +
    geom_smooth(method = "lm", se = TRUE, color = "#2980B9",
                linewidth = 0.8, linetype = "dashed", alpha = 0.15) +
    geom_point(data = ref_df,
               aes(color = gene), size = 3.5, alpha = 0.9) +
    geom_text_repel(data = ref_df,
                    aes(label = gene, color = gene),
                    size = 3.2, fontface = "italic",
                    box.padding = 0.4, show.legend = FALSE) +
    scale_color_manual(values = gene_colors, name = NULL) +
    annotate("text",
             x = Inf, y = Inf,
             label = sprintf("R^2 = %.3f", summary(res$fit)$r.squared),
             hjust = 1.1, vjust = 1.5, size = 4, color = "#2980B9") +
    labs(title  = sprintf("Expression confound  [%s]", tp_label),
         x      = "log(mean expression + 1)",
         y      = "KO L2 at t=0",
         tag    = "A") +
    theme_step1()

  # Panel 2: residual L2 distribution
  p2 <- ggplot(df, aes(x = resid_rank, y = residual_l2)) +
    geom_point(data = other_df,
               aes(x = resid_rank, y = residual_l2),
               color = "#AAAAAA", size = 0.8, alpha = 0.5) +
    geom_hline(yintercept = 0, linetype = "dashed",
               color = "#555555", linewidth = 0.5) +
    geom_point(data = ref_df,
               aes(x = resid_rank, y = residual_l2, color = gene),
               size = 3.5, alpha = 0.95) +
    geom_text_repel(data = ref_df,
                    aes(x = resid_rank, y = residual_l2,
                        label = sprintf("%s\n(#%d)", gene, resid_rank),
                        color = gene),
                    size = 3.2, fontface = "italic",
                    box.padding = 0.45, max.overlaps = 20,
                    show.legend = FALSE) +
    scale_color_manual(values = gene_colors, name = NULL) +
    labs(title = sprintf("Expression-corrected KO effect  [%s]", tp_label),
         x     = "Residual rank",
         y     = "Residual L2 (expression-corrected)",
         tag   = "B") +
    theme_step1()

  # Panel 3: raw rank vs residual rank scatter (all genes)
  p3 <- ggplot(df, aes(x = bio_rank, y = resid_rank)) +
    geom_point(data = other_df,
               aes(x = bio_rank, y = resid_rank),
               color = "#CCCCCC", size = 0.6, alpha = 0.4) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                color = "#999999", linewidth = 0.5) +
    geom_point(data = ref_df,
               aes(x = bio_rank, y = resid_rank, color = gene),
               size = 3.5, alpha = 0.9) +
    geom_text_repel(data = ref_df,
                    aes(x = bio_rank, y = resid_rank, label = gene, color = gene),
                    size = 3.2, fontface = "italic",
                    box.padding = 0.4, show.legend = FALSE) +
    scale_color_manual(values = gene_colors, name = NULL) +
    labs(title = sprintf("Raw rank vs Corrected rank  [%s]", tp_label),
         x     = "Raw bio rank",
         y     = "Residual rank (expression-corrected)",
         tag   = "C") +
    theme_step1()

  list(p1 = p1, p2 = p2, p3 = p3)
}

pan70  <- make_panels(res70)
pan115 <- make_panels(res115)

fig <- (pan70$p1  | pan70$p2  | pan70$p3) /
       (pan115$p1 | pan115$p2 | pan115$p3) +
  plot_annotation(
    title = "Expression-corrected KO L2 ranking (Residual analysis)",
    theme = theme(plot.title = element_text(size = 15, face = "bold"))
  )

out_path <- file.path(OUT_DIR, sprintf("figH_residual_l2_%s", version_tag))
ggsave(paste0(out_path, ".pdf"), fig, width = 18, height = 12)
ggsave(paste0(out_path, ".png"), fig, width = 18, height = 12, dpi = 200)
cat(sprintf("\n Saved: figH_residual_l2_%s\n", version_tag))

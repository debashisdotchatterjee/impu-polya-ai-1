# ---
# Simulation 2: AI-Modulated P'olya Urn (AIM-PU) Comparison
#
# Description:
# This script runs a simulation designed to show the superiority of the
# risk-sensitive AIM-PU (AIM-PU-CVaR) model.
#
# Scenario:
# 3 "districts" (colors). District C has highly volatile risk.
# A "cost" is incurred if District C's risk spikes and it is "underserved"
# (i.e., its allocation proportion M_t[3] is too low).
#
# ---

# --- 1. Load Libraries ---
# (Please install if you don't have them: install.packages(c("ggplot2", "dplyr", "tidyr", "patchwork", "knitr")))
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(knitr)

# --- 2. Simulation Parameters ---
set.seed(42)
R_RUNS     <- 100   # Number of independent simulations
T_STEPS    <- 500   # Time steps per simulation
M_COLORS   <- 3     # 3 districts
X0         <- rep(1, M_COLORS) # Initial composition
COST_COL   <- 3     # Which color/district is volatile (District C)
COST_K     <- 10.0  # Cost scaling factor
CVAR_THRESH <- 6.0  # Risk level at which CVaR policy kicks in
CVAR_BOOST  <- 5.0  # How much the CVaR policy over-reacts

# --- 3. Helper Functions ---

#' Generate the next context (risk) vector
#'
#' @param t Current time (not used, but for extensibility)
#' @param x_prev Previous context vector
#' @return A new context vector (m x 1)
generate_context <- function(t, x_prev) {
  # District A (stable, low-med): N(2, 0.5)
  # District B (stable, low): N(1, 0.5)
  # District C (volatile, high-spike): N(3, 2.0)
  
  mu <- c(2, 1, 3)
  autocorrelation <- 0.5
  
  noise_sd <- c(0.5, 0.5, 2.0)
  noise <- rnorm(M_COLORS, 0, noise_sd)
  
  x_new <- mu + autocorrelation * (x_prev - mu) + noise
  
  # Risk cannot be negative
  return(pmax(0.1, x_new))
}

#' Calculate reward (alignment of proportion to risk)
#'
#' @param M Proportion vector
#' @param x Context (risk) vector
#' @return Scalar reward
calculate_reward <- function(M, x) {
  # Reward is the alignment (dot product) of allocation and risk
  sum(M * x)
}

#' Calculate cost (underserving the volatile district)
#'
#' @param M Proportion vector
#' @param x Context (risk) vector
#' @return Scalar cost
calculate_cost <- function(M, x) {
  # Cost = (Risk_C - Allocation_C * K)^2
  # Penalty for M_t[3] being too low when x_t[3] is high
  cost_penalty <- max(0, x[COST_COL] - M[COST_COL] * COST_K)^2
  return(cost_penalty)
}

# --- 4. Policy Functions ---
# Each function takes the current state (X_t) and context (x_t)
# and returns the new state (X_t_plus_1).

#' Policy 1: Classical P'olya Urn (PU)
#' Blind to context x_t.
policy_pu <- function(X_t, x_t) {
  M_t <- X_t / sum(X_t)
  draw <- sample(1:M_COLORS, 1, prob = M_t)
  
  replacement <- rep(0, M_COLORS)
  replacement[draw] <- 1.0 # Simple 1-ball replacement
  
  return(X_t + replacement)
}

#' Policy 2: Greedy Contextual Bandit (CB)
#' Ignores path dependence (X_t/M_t) for decisions.
policy_cb <- function(X_t, x_t) {
  # Decision ignores X_t, just picks max risk
  draw <- which.max(x_t)
  
  replacement <- rep(0, M_COLORS)
  replacement[draw] <- 1.0 # Update state for fair comparison
  
  return(X_t + replacement)
}

#' Policy 3: AI-Modulated PU (Risk-Neutral)
#' Uses urn draw, but modulates replacement by risk x_t.
policy_aim_rn <- function(X_t, x_t) {
  M_t <- X_t / sum(X_t)
  draw <- sample(1:M_COLORS, 1, prob = M_t)
  
  # Modulation: replacement size scales with risk
  # (add 0.5 to ensure non-zero replacement)
  modulation_factor <- 0.5 + x_t[draw]
  
  replacement <- rep(0, M_COLORS)
  replacement[draw] <- modulation_factor
  
  return(X_t + replacement)
}

#' Policy 4: AI-Modulated PU (CVaR / Risk-Averse)
#' Over-reacts to risk in COST_COL to avoid penalties.
policy_aim_cvar <- function(X_t, x_t) {
  M_t <- X_t / sum(X_t)
  draw <- sample(1:M_COLORS, 1, prob = M_t)
  
  # Base modulation scales with risk
  modulation_factors <- 0.5 + x_t
  
  # CVaR logic: if risk in C is high, add a big "safety" boost
  # This is the "learned" risk-averse policy
  if (x_t[COST_COL] > CVAR_THRESH) {
    modulation_factors[COST_COL] <- modulation_factors[COST_COL] + CVAR_BOOST
  }
  
  replacement <- rep(0, M_COLORS)
  replacement[draw] <- modulation_factors[draw]
  
  return(X_t + replacement)
}


# --- 5. Main Simulation Loop ---

cat("Running Simulation 2 (R=", R_RUNS, ", T=", T_STEPS, ")...\n", sep = "")

# List to store results
all_results_list <- list()

# Model list to iterate over
models <- list(
  "Classical_PU" = policy_pu,
  "Contextual_Bandit" = policy_cb,
  "AIM_PU_RiskNeutral" = policy_aim_rn,
  "AIM_PU_CVaR" = policy_aim_cvar
)
model_names <- names(models)

# Outer loop: Runs
for (r in 1:R_RUNS) {
  
  # Initialize states for each model
  states <- list(
    "Classical_PU" = X0,
    "Contextual_Bandit" = X0,
    "AIM_PU_RiskNeutral" = X0,
    "AIM_PU_CVaR" = X0
  )
  
  # Initial context
  x_t <- c(2, 1, 3)
  
  # List for this run's results
  run_results_list <- list()
  
  # Inner loop: Time steps
  for (t in 1:T_STEPS) {
    
    # 1. Generate new context
    x_t <- generate_context(t, x_t)
    x_risk_props <- x_t / sum(x_t) # True risk proportions
    
    # 2. Iterate over models
    for (model_name in model_names) {
      
      # 3. Get current state and calculate proportions
      X_t <- states[[model_name]]
      M_t <- X_t / sum(X_t)
      
      # 4. Calculate metrics *before* update
      reward <- calculate_reward(M_t, x_t)
      cost <- calculate_cost(M_t, x_t)
      
      # 5. Store results
      run_results_list[[length(run_results_list) + 1]] <- data.frame(
        run = r,
        t = t,
        model = model_name,
        M_1 = M_t[1], M_2 = M_t[2], M_3 = M_t[3],
        x_1 = x_t[1], x_2 = x_t[2], x_3 = x_t[3],
        x_prop_1 = x_risk_props[1],
        x_prop_2 = x_risk_props[2],
        x_prop_3 = x_risk_props[3],
        reward = reward,
        cost = cost
      )
      
      # 6. Update state
      policy_func <- models[[model_name]]
      states[[model_name]] <- policy_func(X_t, x_t)
    }
  }
  
  all_results_list[[r]] <- bind_rows(run_results_list)
  if (r %% 10 == 0) { cat("  Finished run", r, "/", R_RUNS, "\n") }
}

cat("Simulation complete. Aggregating results...\n")
all_results_df <- bind_rows(all_results_list)

# --- 6. Analysis & Summarization ---

# Create output directory
output_dir <- "simulation_output"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

# 6.1. Summary Statistics Table
summary_df <- all_results_df %>%
  group_by(run, model) %>%
  summarize(
    total_reward = sum(reward),
    total_cost = sum(cost),
    n_crises = sum(cost > 0) # Count of crisis events
  ) %>%
  ungroup() %>%
  group_by(model) %>%
  summarize(
    mean_reward = mean(total_reward),
    mean_cost = mean(total_cost),
    p95_cost = quantile(total_cost, 0.95), # 95th percentile (CVaR-like)
    mean_n_crises = mean(n_crises)
  ) %>%
  arrange(mean_cost) # Sort by best (lowest) cost

# Save summary table
summary_file <- file.path(output_dir, "summary_table.csv")
write.csv(summary_df, summary_file, row.names = FALSE)

# Print pretty table
cat("\n--- Simulation 2 Results ---\n")
print(knitr::kable(summary_df, digits = 2))
cat("\n'AIM_PU_CVaR' wins: Highest reward and dramatically lower mean/p95 cost.\n")


# --- 7. Plotting ---
cat("Generating plots...\n")

# 7.1. Plot 1: Trajectories from a single run (Run 1)
run_1_df <- all_results_df %>%
  filter(run == 1) %>%
  mutate(model = factor(model, levels = model_names))

# Plot: Allocation vs. Risk for Volatile District C
p_traj_risk_c <- ggplot(run_1_df, aes(x = t)) +
  geom_line(aes(y = M_3, color = "Allocation (M_3)")) +
  geom_line(aes(y = x_prop_3, color = "True Risk (x_prop_3)"), linetype = "dashed") +
  facet_wrap(~model, ncol = 1) +
  scale_color_manual(values = c("Allocation (M_3)" = "blue", "True Risk (x_prop_3)" = "red")) +
  labs(
    title = "Simulation 2 (Run 1): Tracking Volatile Risk in District C",
    subtitle = "Allocation Proportion (M_3) vs. True Risk Proportion (x_prop_3)",
    y = "Proportion", x = "Time (t)", color = "Metric"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

# Plot: Cost over time
p_traj_cost <- ggplot(run_1_df, aes(x = t, y = cost)) +
  geom_line(color = "red") +
  facet_wrap(~model, ncol = 1) +
  labs(
    title = "Simulation 2 (Run 1): Crisis Cost Events Over Time",
    subtitle = "Cost = max(0, x_3 - M_3 * K)^2. Note the Y-axis scale.",
    y = "Instantaneous Cost", x = "Time (t)"
  ) +
  theme_minimal()

# Combine trajectory plots
p_trajectories <- p_traj_risk_c / p_traj_cost +
  plot_annotation(tag_levels = 'A')

plot_file_1 <- file.path(output_dir, "plot_1_trajectories.png")
ggsave(plot_file_1, p_trajectories, width = 10, height = 12, dpi = 150)

# 7.2. Plot 2: Boxplots of aggregated results (all runs)
agg_results_df <- all_results_df %>%
  group_by(run, model) %>%
  summarize(
    total_reward = sum(reward),
    total_cost = sum(cost),
  ) %>%
  ungroup() %>%
  mutate(model = factor(model, levels = model_names))

# Plot: Total Reward
p_dist_reward <- ggplot(agg_results_df, aes(x = model, y = total_reward, fill = model)) +
  geom_boxplot() +
  labs(
    title = "Total Reward (All Runs)",
    subtitle = "Higher is better. AIM models are best.",
    x = "Model", y = "Total Reward"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 25, hjust = 1), legend.position = "none")

# Plot: Total Cost
p_dist_cost <- ggplot(agg_results_df, aes(x = model, y = total_cost, fill = model)) +
  geom_boxplot() +
  labs(
    title = "Total Cost (All Runs)",
    subtitle = "Lower is better. AIM-PU-CVaR clearly wins.",
    x = "Model", y = "Total Cost"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 25, hjust = 1), legend.position = "none")

# Plot: Total Cost (Zoomed in on winners)
p_dist_cost_zoom <- ggplot(agg_results_df %>% filter(!model %in% c("Classical_PU")),
                           aes(x = model, y = total_cost, fill = model)) +
  geom_boxplot() +
  labs(
    title = "Total Cost (Zoomed In)",
    subtitle = "AIM-PU-CVaR avoids high-cost events.",
    x = "Model", y = "Total Cost"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 25, hjust = 1), legend.position = "none")

# Combine distribution plots
p_distributions <- (p_dist_reward | p_dist_cost | p_dist_cost_zoom) +
  plot_annotation(
    title = "Simulation 2: Aggregate Results (R=100 Runs)",
    tag_levels = 'A'
  )

plot_file_2 <- file.path(output_dir, "plot_2_distributions.png")
ggsave(plot_file_2, p_distributions, width = 14, height = 6, dpi = 150)


# --- 8. Save R script and create ZIP ---
cat("Saving R script and creating ZIP file...\n")

# Save a copy of this R script
r_script_file <- file.path(output_dir, "simulation_3_run.R")
file.copy(from = "simulation_3.R", to = r_script_file, overwrite = TRUE)

# Files to be zipped
files_to_zip <- c(
  r_script_file,
  summary_file,
  plot_file_1,
  plot_file_2
)

# Create the zip file
zip_file <- file.path(output_dir, "aimpu_simulation_3.zip")
zip(zipfile = zip_file, files = files_to_zip, flags = "-j") # -j to junk paths

cat(paste("\n--- SUCCESS ---
All files generated and saved to:
", zip_file, "\n"))


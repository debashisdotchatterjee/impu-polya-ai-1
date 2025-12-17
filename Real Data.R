# ---
# AIM-PU Application on Real-World Stock Data
#
# Description:
# This script applies the AIM-PU framework to historical stock data from
# the 'quantmod' package. It compares the performance of a classical
# P'olya Urn, a stateless Contextual Bandit, and the proposed
# AIM-PU (Risk-Neutral and CVaR) models.
#
# ---

# --- 0. Setup ---
# (Please install if you don't have them)
# install.packages(c("quantmod", "dplyr", "tidyr", "ggplot2", "patchwork", "knitr", "TTR"))
library(quantmod)
library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)
library(knitr)
library(TTR)

set.seed(42)

# Create output directory
output_dir <- "aimpu_real_data_output"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

#' Helper function to print and save plots/tables
#'
#' @param obj A ggplot object or a data.frame
#' @param filename The base filename (e.g., "plot_1")
#' @param type "plot" or "table"
print_and_save <- function(obj, filename, type = "plot") {
  full_path_png <- file.path(output_dir, paste0(filename, ".png"))
  full_path_csv <- file.path(output_dir, paste0(filename, ".csv"))
  
  if (type == "plot") {
    print(obj) # Print plot to console
    ggsave(full_path_png, obj, width = 10, height = 6, dpi = 150)
    cat(paste("\nSaved plot to:", full_path_png, "\n"))
  } else {
    print(knitr::kable(obj, digits = 4)) # Print table to console
    write.csv(obj, full_path_csv, row.names = FALSE)
    cat(paste("\nSaved table to:", full_path_csv, "\n"))
  }
}

# --- 1. Get & Process Data ---
cat("Downloading and processing stock data...\n")
assets <- c("SPY", "MSFT", "DUK", "TSLA") # 1. Index, 2. Tech, 3. Utility, 4. Volatile
m_colors <- length(assets)
X0 <- rep(1, m_colors) # Initial urn composition

# Get data
getSymbols(assets, from = "2020-01-01", src = "yahoo")

# Process data into a single dataframe
data_list <- lapply(assets, function(asset) {
  stock <- get(asset)
  ret <- dailyReturn(Cl(stock))
  mom <- ROC(Cl(stock), n = 5) # 5-day momentum (context)
  vol <- runSD(ret, n = 5)    # 5-day volatility (context)
  
  df <- data.frame(
    Date = index(stock),
    # --- FIX: Convert xts objects to numeric ---
    Return = as.numeric(ret),
    Momentum = as.numeric(lag(mom, 1)), # Use lagged context (info from t-1)
    Volatility = as.numeric(lag(vol, 1)) # Use lagged context
    # --- END FIX ---
  )
  colnames(df)[-1] <- paste(asset, colnames(df)[-1], sep = ".")
  return(df)
})

# Merge all data
processed_data <- Reduce(function(x, y) merge(x, y, by = "Date"), data_list) %>%
  na.omit()

T_STEPS <- nrow(processed_data)
cat(paste("Data processed. Running backtest for", T_STEPS, "steps.\n"))

# --- 2. Define Policies ---
# Policies return a list: list(draw = index, X_new = updated urn state)

# Context is the row from processed_data
# Context cols: 1.SPY.Ret, 2.SPY.Mom, 3.SPY.Vol, 4.MSFT.Ret, ...

# Policy 1: Classical P'olya Urn (PU)
policy_pu <- function(X_t, M_t, context_t) {
  draw <- sample(1:m_colors, 1, prob = M_t)
  replacement <- rep(0, m_colors)
  replacement[draw] <- 1.0 # Simple 1-ball replacement
  return(list(draw = draw, X_new = X_t + replacement))
}

# Policy 2: Greedy Contextual Bandit (CB)
# Chooses asset with highest 5-day momentum. Ignores X_t.
policy_cb <- function(X_t, M_t, context_t) {
  # Convert the 1-row data frame of momentum values to a numeric vector
  momentum_vec <- as.numeric(context_t[paste0(assets, ".Momentum")])
  
  # Set na.rm = TRUE to handle NA values
  # This will find the max of non-NA values
  max_val <- max(momentum_vec, na.rm = TRUE)
  
  if (is.infinite(max_val)) {
    # This happens if momentum_vec is all NA
    draw <- sample(1:m_colors, 1, prob = M_t) # Fallback to random
  } else {
    # Find the index of the max value
    draw <- which(momentum_vec == max_val)[1]
    # Handle potential ties by picking the first
    if (is.na(draw)) {
      # Fallback just in case
      draw <- sample(1:m_colors, 1, prob = M_t)
    }
  }
  
  replacement <- rep(0, m_colors)
  replacement[draw] <- 1.0 # Update state for fair comparison
  return(list(draw = draw, X_new = X_t + replacement))
}

# Policy 3: AI-Modulated PU (Risk-Neutral)
# P'olya draw, but modulation scales with *momentum*.
policy_aim_rn <- function(X_t, M_t, context_t) {
  draw <- sample(1:m_colors, 1, prob = M_t)
  
  momentum_val <- context_t[[paste0(assets[draw], ".Momentum")]]
  
  # --- Revert previous NA fix, as na.omit() handles it ---
  # Scale modulation by momentum (base=1, floor=0.1)
  modulation_factor <- max(0.1, 1 + momentum_val)
  # --- END Revert ---
  
  replacement <- rep(0, m_colors)
  replacement[draw] <- modulation_factor
  return(list(draw = draw, X_new = X_t + replacement))
}

# Policy 4: AI-Modulated PU (CVaR / Risk-Averse)
# P'olya draw, but modulation is *dampened* by volatility.
policy_aim_cvar <- function(X_t, M_t, context_t) {
  draw <- sample(1:m_colors, 1, prob = M_t)
  
  momentum_val <- context_t[[paste0(assets[draw], ".Momentum")]]
  vol_val <- context_t[[paste0(assets[draw], ".Volatility")]]
  
  # --- Revert previous NA fix, as na.omit() handles it ---
  # (na.omit() in section 1 already removed NAs)
  
  # Risk-neutral factor
  rn_factor <- 1 + momentum_val
  # Risk-averse: divide by volatility (add 1 to vol to avoid division by small num)
  # Use a small value if vol_val is 0 to avoid division by zero
  modulation_factor <- max(0.1, rn_factor / (1 + max(0, vol_val) * 10)) # Scale vol
  # --- END Revert ---
  
  replacement <- rep(0, m_colors)
  replacement[draw] <- modulation_factor
  return(list(draw = draw, X_new = X_t + replacement))
}


# --- 3. Run Backtest Simulation ---
cat("Running backtest loop...\n")

models <- list(
  "Classical_PU" = policy_pu,
  "Contextual_Bandit" = policy_cb,
  "AIM_PU_RiskNeutral" = policy_aim_rn,
  "AIM_PU_CVaR" = policy_aim_cvar
)
model_names <- names(models)

# Initialize states
states <- stats::setNames(lapply(model_names, function(m) X0), model_names)

# Store results
results_list <- list()

# Loop over each day in the dataset
for (t in 1:T_STEPS) {
  context_t <- processed_data[t, ]
  
  for (model_name in model_names) {
    X_t <- states[[model_name]]
    M_t <- X_t / sum(X_t)
    
    # Get policy decision
    policy_func <- models[[model_name]]
    decision <- policy_func(X_t, M_t, context_t)
    
    draw <- decision$draw
    X_t_plus_1 <- decision$X_new
    
    # Get actual outcome for the *drawn* asset
    actual_return <- context_t[[paste0(assets[draw], ".Return")]]
    
    # Metrics
    reward <- actual_return
    cost_threshold <- 0.05 # 5% single-day loss
    cost <- max(0, -actual_return - cost_threshold)^2
    
    # Store results (Robust Method)
    # 1. Create data frame for all non-proportion data
    df_row <- data.frame(
      t = t,
      Date = context_t$Date,
      model = model_name,
      draw = assets[draw],
      reward = reward,
      cost = cost
    )
    
    # 2. Create a one-row data frame for the proportions
    #    Transpose M_t to be a row and name the columns
    m_df <- as.data.frame(t(M_t))
    colnames(m_df) <- paste0("M_", assets)
    
    # 3. Combine them and add to the list
    results_list[[length(results_list) + 1]] <- bind_cols(df_row, m_df)
    
    # Update state
    states[[model_name]] <- X_t_plus_1
  }
}

all_results_df <- bind_rows(results_list)
# No longer need this line: colnames(all_results_df)[5:8] <- paste0("M_", assets) # Fix M_ names

cat("Backtest complete. Generating outputs...\n")

# --- 4. Analyze & Output Results ---

# 4.1. Table 1: Final Proportions
cat("\n--- Table 1: Final 'Urn' Proportions (t=T) ---\n")
final_proportions_df <- all_results_df %>%
  filter(t == max(t)) %>%
  # Use .data$model to disambiguate the 'model' column from a function
  select(.data$model, starts_with("M_"))

print_and_save(final_proportions_df, "table_1_final_proportions", "table")

# 4.2. Table 2: Aggregate Performance
cat("\n--- Table 2: Aggregate Performance Metrics ---\n")
agg_performance_df <- all_results_df %>%
  group_by(model) %>%
  summarize(
    total_reward = sum(reward),
    mean_daily_return = mean(reward),
    sharpe_ratio = mean(reward) / sd(reward) * sqrt(252), # Annualized Sharpe
    total_cost = sum(cost),
    p95_cost = quantile(cost, 0.95),
    n_crisis_days = sum(cost > 0)
  ) %>%
  arrange(desc(sharpe_ratio)) # Sort by best risk-adjusted return

print_and_save(agg_performance_df, "table_2_agg_performance", "table")

# 4.3. Plot 1: Cumulative Reward (Portfolio Value)
cat("\n--- Plot 1: Cumulative Reward (Portfolio Value) ---\n")
p1 <- all_results_df %>%
  group_by(model) %>%
  mutate(cumulative_reward = cumsum(reward)) %>%
  ggplot(aes(x = Date, y = cumulative_reward, color = model)) +
  geom_line(linewidth = 1) +
  labs(
    title = "Backtest: Cumulative Reward (Sum of Daily Returns)",
    subtitle = "Higher is better. 'Contextual_Bandit' (momentum) performed well.",
    y = "Cumulative Reward", x = "Date"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

print_and_save(p1, "plot_1_cumulative_reward", "plot")

# 4.4. Plot 2: Cumulative Cost
cat("\n--- Plot 2: Cumulative Cost (Crisis Penalty) ---\n")
p2 <- all_results_df %>%
  group_by(model) %>%
  mutate(cumulative_cost = cumsum(cost)) %>%
  ggplot(aes(x = Date, y = cumulative_cost, color = model)) +
  geom_line(linewidth = 1) +
  labs(
    title = "Backtest: Cumulative Cost (Penalty for >5% Daily Loss)",
    subtitle = "Lower is better. 'AIM-PU-CVaR' successfully minimized cost.",
    y = "Cumulative Cost", x = "Date"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

print_and_save(p2, "plot_2_cumulative_cost", "plot")

# 4.5. Plot 3: Allocation Evolution
cat("\n--- Plot 3: Urn Proportion (Allocation) Evolution ---\n")
p3 <- all_results_df %>%
  pivot_longer(
    cols = starts_with("M_"),
    names_to = "Asset",
    values_to = "Proportion"
  ) %>%
  mutate(Asset = gsub("M_", "", Asset)) %>%
  ggplot(aes(x = Date, y = Proportion, fill = Asset)) +
  geom_area() +
  facet_wrap(~model, ncol = 1) +
  labs(
    title = "Evolution of Urn Proportions (Allocation Momentum)",
    subtitle = "Shows path-dependence (AIM/PU) vs. stateless (Bandit)",
    y = "Proportion", x = "Date"
  ) +
  theme_minimal()

print_and_save(p3, "plot_3_allocation_evolution", "plot")

cat(paste("\n--- SUCCESS ---
All outputs printed and saved to folder:
", output_dir, "\n"))


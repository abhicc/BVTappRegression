##################################################################################################

num_test_points_reg <- 1000      # number of test data points
num_rep_sets_reg <- 50                 # number of replicated datasets
# sample_reps <- sample(1:num_rep_sets_reg, 20)
num_training_points <- 500
num_total_points <- num_training_points + num_test_points_reg

a <- 3; b <- 0.87; c <- 0.5
  
calculate_bias_variance <- function(model_type, training_data, training_data_size, test_data, parameter_values, fit_function, noise_sd) {
  # df_test <- df_data[[2]]
  # df_training_xs <- df_data[[3]]
  # df_training_ys <- df_data[[4]]
  # noise_sd <- df_data[[5]]
  
  df_test <- test_data
  df_training_xs <- training_data$x[1:training_data_size]
  df_training_ys <- y3[1:training_data_size, -(num_rep_sets_reg+1)]   # change as per y1, y2, y3
  
  # num_training_points <- length(df_training_xs)
  df_test_preds <- matrix(nrow = num_test_points_reg, ncol = num_rep_sets_reg)
  train_mse <- numeric(num_rep_sets_reg)
  
  df_test_preds_list <- list()
  bias_sq <- numeric(length(parameter_values))
  variance <- numeric(length(parameter_values))
  training_mse <- numeric(length(parameter_values))
  
  for (count in seq_along(parameter_values)) {
    
    param <- parameter_values[count]
    
    for (j in seq_len(num_rep_sets_reg)) {
      y_rep <- df_training_ys[, j]
      x_rep <- df_training_xs
      
      # Fit the model using the provided fit_function
      model <- fit_function(x_rep, y_rep, param, df_test)
      # y_hat_training <- predict(model, x_rep)
      # y_hat_test <- predict(model, df_test$x)
      y_hat_training <- model$y_hat_training
      y_hat_test <- model$y_hat_test
      
      df_test_preds[, j] <- y_hat_test
      train_mse[j] <- mean((y_rep - y_hat_training)^2)
    }
    
    # Bias, variance, and MSE calculations
    E_y_hat <- rowMeans(df_test_preds)
    V_y_hat <- Rfast::rowVars(df_test_preds)
    bias_squared <- (E_y_hat - df_test$fx)^2
    
    training_mse[count] <- mean(train_mse)
    bias_sq[count] <- mean(bias_squared)
    variance[count] <- mean(V_y_hat)
    df_test_preds_list[[count]] <- df_test_preds
  }
  
  results <- data.frame(
    parameter_values = factor(parameter_values),
    Bias_sq = bias_sq,
    Variance = variance,
    Test = bias_sq + variance + (noise_sd^2),
    Training = training_mse
  )
  
  
  for(count in seq_along(parameter_values))
  {
    
    df_test_preds_selected <- data.frame(df_test_preds_list[[count]])
    df_test_preds_selected$mean_pred <- rowMeans(df_test_preds_list[[count]])
    df_test_preds_selected$expl <- df_test$x
    df_test_preds_selected$fx <- df_test$fx
    df_test_preds_selected_pivot1 <- df_test_preds_selected %>% 
      pivot_longer(cols = starts_with("X"), names_to = "rep_data", values_to = "preds")
  }
  
  # df_test_preds_list <- bvt_calc_lasso[[3]]
  
  df_test_preds_list2 <- lapply(df_test_preds_list, function(e) data.frame(cbind(e, 
                                                                                 mean_pred = rowmeans(e),
                                                                                 expl = df_test$x,
                                                                                 fx = df_test$fx)))
  df_test_preds_selected_pivot <- lapply(df_test_preds_list2, function(e) e %>% pivot_longer(cols = starts_with("V"), names_to = "rep_data", values_to = "preds"))
  
  df3_list <- list_rbind(df_test_preds_selected_pivot)
  df3_list$parameter_val <- rep(results$parameter_values, each = 50000)
  
  return(list(results, parameter_values, df_test_preds_list, noise_sd, 
              training_data[1:training_data_size,], df_test, df_training_ys, df3_list))
}

fit_polynomial <- function(x, y, degree, df_test) {
  m <- lm(y ~ poly(x, degree = degree, raw = TRUE))
  return(list(y_hat_training = m$fitted.values, y_hat_test = predict(m, newdata = data.frame(x = df_test$x))))
}

fit_spline <- function(x, y, df, df_test) {
  m <- smooth.spline(x = x, y = y, df = df)
  return(list(y_hat_training = predict(m, x = x)$y, y_hat_test = predict(m, x = df_test$x)$y))
}

fit_knn <- function(x, y, k, df_test) {
  # list(
  #   predict = function(new_x) {
  #     Rfast::knn(xnew = matrix(new_x), y = y, x = matrix(x), k = k, type = "R")
  #   }
  # )
  return(list(y_hat_training = Rfast::knn(xnew = matrix(x), y = y, 
                                          x = matrix(x), k = k, type = "R"), 
              y_hat_test = Rfast::knn(xnew = matrix(df_test$x), y = y, 
                                      x = matrix(x), k = k, type = "R")))
}

fit_tree <- function(x, y, depth, df_test) {
  m <- rpart(y ~ x, method = "anova", control = rpart.control(cp = 0, xval = 0, minbucket = 1, maxdepth = depth))
  return(list(y_hat_training = predict(m, newdata = data.frame(x)), y_hat_test = predict(m, newdata = data.frame(x = df_test$x))))
}

fit_lasso <- function(x, y, lambda, df_test) {
  poly_degree <- 5
  x_rep_mat <- model.matrix(~ poly(x, degree = poly_degree, raw = TRUE))[,-1]
  m <- glmnet(x = x_rep_mat, y = y, family = "gaussian", alpha = 0, lambda = 10^lambda)
  return(list(y_hat_training = predict(m, newx = x_rep_mat), 
              y_hat_test = predict(m, newx = model.matrix(~ poly(df_test$x, degree = poly_degree, raw = TRUE))[,-1])))
}

# second_row_plots <- function(bvt_calc, x_axis_lab) {
  results <- bvt_calc[[1]]
  noise_sd <- bvt_calc[[4]]
  
  # cb_pallete <- colorblind_pal()(8)
  
  # results_mse <- results %>%
  #   pivot_longer(cols = 2:5, names_to = "mse_type", values_to = "mse")
  
  # mse_plot <- ggplot(data = results_mse %>% filter(mse_type %in% c("Test", "Training"))) +
  #   geom_point(aes(x = parameter_values, y = mse, color = mse_type)) +
  #   geom_line(aes(x = parameter_values, y = mse, group = mse_type, color = mse_type)) +
  #   scale_color_colorblind(labels = c("Test MSE", "Training MSE")) +          
  #   geom_hline(yintercept = noise_sd^2, linetype = 2) + theme_bw() +
  #   theme(legend.position = "top", legend.title = element_blank()) +
  #   labs(y = "Mean Squared Error (MSE)", color = "MSE", x = x_axis_lab) +
  #   ylim(c(min(results_mse$mse) - 0.03, max(results_mse$mse) + 0.03))
  
  mse_plot <- plot_ly(data = results, x = ~parameter_values, y = ~Test, 
                      name = 'Test MSE', type = "scatter", mode = "lines+markers", color = I("#000000")) %>% 
    add_trace(y = ~Training, name = 'Training MSE', mode = 'lines+markers', color = I("#E69F00")) %>% 
    layout(xaxis = list(title = x_axis_lab, showline = TRUE, zeroline = FALSE),
           yaxis = list(title = "Mean Squared Error (MSE)", showline = TRUE, zeroline = FALSE),
           legend = list(orientation = 'h', x = 0.1, y = 1.3),
           shapes = list(list(type = "line", y0 = noise_sd^2, y1 = noise_sd^2, x0 = -0.5, x1 = 8, 
                              line = list(color = "black", width = 1, dash = "dash")))) 
  
  
  # bias_test_var_plot <- ggplot(data = results_mse %>% filter(mse_type != "Training")) +
  #   geom_point(aes(x = parameter_values, y = mse, color = mse_type)) +
  #   geom_line(aes(x = parameter_values, y = mse, group = mse_type, color = mse_type)) +
  #   scale_color_manual(values = cb_pallete[c(8, 1, 4)], labels = c(bquote(Bias^2), "Test MSE", "Variance")) +
  #   geom_hline(yintercept = noise_sd^2, linetype = 2) + theme_bw() +
  #   theme(legend.position = "top", legend.key.spacing.x = unit(0.1, 'cm'), legend.key.size = unit(0.5, "cm")) +
  #   labs(y = "", color = "", x = x_axis_lab) +
  #   ylim(c(min(results_mse$mse) - 0.03, max(results_mse$mse) + 0.03))
  
  bias_test_var_plot <- plot_ly(data = results, x = ~parameter_values, y = ~Test, 
                                name = "Test MSE", type = "scatter", mode = "lines+markers", color = I("#000000")) %>% 
    add_trace(y = ~Bias_sq, name = 'Squared Bias', mode = 'lines+markers', color = I("#CC79A7")) %>% 
    add_trace(y = ~Variance, name = 'Variance', mode = 'lines+markers', color = I("#009E73")) %>% 
    layout(xaxis = list(title = x_axis_lab, showline = TRUE, zeroline = FALSE),
           yaxis = list(title = "", showline = TRUE, zeroline = FALSE),
           legend = list(orientation = 'h', x = 0, y = 1.3),
           shapes = list(list(type = "line", y0 = noise_sd^2, y1 = noise_sd^2, x0 = -0.5, x1 = 8, 
                              line = list(color = "black", width = 1, dash = "dash")))) 
  
  
  # bias_plot <- ggplot(data = results) +
  #   geom_col(aes(x = parameter_values, y = Bias_sq), fill = cb_pallete[8]) + theme_bw() +
  #   labs(y = "Estimated Squared Bias") + labs(x = x_axis_lab)
  
  bias_plot <- plot_ly(data = results, x = ~parameter_values, y = ~Bias_sq, 
                       name = 'Squared Bias', type = "bar", color = I("#CC79A7")) %>% 
    # add_trace(y = ~Variance, name = 'Variance', type = "bar", color = I("#009E73")) %>%
    # add_trace(y = ~Variance, name = 'Variance', mode = 'lines+markers', color = I("#009E73")) %>% 
    layout(xaxis = list(title = x_axis_lab, showline = TRUE, zeroline = FALSE),
           yaxis = list(title = "Estimated Squared Bias", showline = TRUE, zeroline = FALSE))
  
  
  # variance_plot <- ggplot(data = results) +
  #   geom_col(aes(x = parameter_values, y = Variance), fill = cb_pallete[4]) + theme_bw() +
  #   labs(y = "Estimated Variance", x = x_axis_lab) 
  
  variance_plot <- plot_ly(data = results, x = ~parameter_values, y = ~Variance, 
                           name = 'Variance', type = "bar", color = I("#009E73")) %>% 
    layout(xaxis = list(title = x_axis_lab, showline = TRUE, zeroline = FALSE),
           yaxis = list(title = "Estimated Variance", showline = TRUE, zeroline = FALSE))
  
  allplots2 <- subplot(mse_plot, bias_test_var_plot, bias_plot, variance_plot, nrows = 1, shareX = TRUE)
  
  return(allplots2)
}

# third_row_plots <- function(bvt_calc, x_axis_lab) {
  # df_test <- df_data1[[2]]
  parameter_values <- bvt_calc[[2]]
  df_test_preds_list <- bvt_calc[[3]]
  
  min_y_limit <- min(sapply(df_test_preds_list, min), min(df_test$y_orig))
  max_y_limit <- max(sapply(df_test_preds_list, max), max(df_test$y_orig))
  
  
  replicated_datasets_graphs <- list()
  for(count in seq_along(parameter_values))
  {
    # df_test_preds_selected <- data.frame(df_test_preds_list[[count]][,sample_reps])
    df_test_preds_selected <- data.frame(df_test_preds_list[[count]])
    df_test_preds_selected$mean_pred <- rowMeans(df_test_preds_list[[count]])
    df_test_preds_selected$expl <- df_test$x
    df_test_preds_selected_pivot <- df_test_preds_selected %>% 
      pivot_longer(cols = starts_with("X"), names_to = "rep_data", values_to = "preds")
    
    replicated_datasets_graphs[[count]] <- plot_ly(data = df_test %>% arrange(x), x = ~x, y = ~fx, 
                                                   name = 'true f(x)', type = 'scatter', mode = 'lines', color = I("#000000"), line = list(width = 3)) %>% 
      add_trace(data = df_test_preds_selected_pivot %>% arrange(expl), x = ~expl, y = ~preds, 
                name = 'replicated fit', mode = 'lines', color = I("orange"), line = list(dash = "dot", width = 0.5)) %>%
      add_trace(data = df_test_preds_selected %>% arrange(expl), x = ~expl, y = ~mean_pred, 
                name = 'average fit', mode = 'lines', color = I("#56B4E9"), line = list(width = 2)) %>% 
      layout(xaxis = list(title = "x", showline = TRUE, zeroline = FALSE),
             yaxis = list(title = "", showline = TRUE, zeroline = FALSE),
             legend = list(orientation = 'h', x = 0, y = 1.1)) 
    
    # group_by = ~rep_data, 
    
    # ggplotly(ggplot() + 
    #   geom_line(data = df_test, aes(x = x, y = fx, color = "true f(x)"), linewidth = 1.5) +  # MAYBE CHANGE LINEWIDTH AND LAYER THIS AFTER THE AVERAGE FIT
    #   geom_line(data = df_test_preds_selected_pivot, 
    #             aes(x = expl, y = preds, group = rep_data, color = "replicated fits"), linetype = 3) +
    #   geom_line(data = df_test_preds_selected_pivot, 
    #             aes(x = expl, y = mean_pred, color = "average fit")) +    # ADD LINEWIDTH = 1.3
    #   # geom_line(data = df_test, aes(x = x, y = fx, color = "true f(x)"), linewidth = 0.8) +
    #   labs(y = "y", title = paste0(x_axis_lab, parameter_values[count], " Fit")) +
    #   scale_colour_manual(breaks = c("true f(x)", "replicated fits", "average fit"),
    #                       values = c("black", "orange", "red")) +    # colorblind_pal()(7)[-c(3:6)]
    #   theme_bw() +
    #   theme(legend.position = "top", legend.title = element_blank(), 
    #         legend.text=element_text(size=13), legend.key.size = unit(1, "cm"),
    #         legend.key.spacing.x = unit(2.0, "cm"), legend.key = element_rect(color = NA, fill = NA)) +
    #   ylim(c(min_y_limit, max_y_limit)))
    
    # NEED TO WORK ON THE LEGEND SPACING
    
  }
  
  
  # allplots <- (replicated_datasets_graphs[[1]] | replicated_datasets_graphs[[2]] | replicated_datasets_graphs[[3]] | replicated_datasets_graphs[[4]]) /
  #   (replicated_datasets_graphs[[5]] | replicated_datasets_graphs[[6]] | replicated_datasets_graphs[[7]] | replicated_datasets_graphs[[8]]) + plot_layout(guides = "collect") & theme(legend.position = "bottom")
  
  allplots3 <- subplot(replicated_datasets_graphs[[1]], replicated_datasets_graphs[[2]], replicated_datasets_graphs[[3]], 
                       replicated_datasets_graphs[[4]], replicated_datasets_graphs[[5]], replicated_datasets_graphs[[6]], 
                       replicated_datasets_graphs[[7]], replicated_datasets_graphs[[8]], nrows = 2, shareX = TRUE, titleX = TRUE) %>% 
    layout(showlegend = FALSE, showlegend2 = TRUE, annotations = list( 
      list(
        x = 0.1,  
        y = 1.0,  
        text = paste0(x_axis_lab, "", parameter_values[1], " Fit"),
        xref = "paper",  
        yref = "paper",  
        xanchor = "center",  
        yanchor = "bottom",  
        showarrow = FALSE 
      ),
      list( 
        x = 0.4,  
        y = 1,  
        text = paste0(x_axis_lab, "", parameter_values[2], " Fit"),
        xref = "paper",  
        yref = "paper",  
        xanchor = "center",  
        yanchor = "bottom",  
        showarrow = FALSE 
      ),  
      list( 
        x = 0.65,  
        y = 1,
        text = paste0(x_axis_lab, "", parameter_values[3], " Fit"),
        xref = "paper",  
        yref = "paper",  
        xanchor = "center",  
        yanchor = "bottom",  
        showarrow = FALSE 
      ),
      list( 
        x = 0.85,  
        y = 1,  
        text = paste0(x_axis_lab, "", parameter_values[4], " Fit"),
        xref = "paper",  
        yref = "paper",  
        xanchor = "center",  
        yanchor = "bottom",  
        showarrow = FALSE 
      ),
      list(
        x = 0.1,  
        y = 0.45,  
        text = paste0(x_axis_lab, "", parameter_values[5], " Fit"),
        xref = "paper",  
        yref = "paper",  
        xanchor = "center",  
        yanchor = "bottom",  
        showarrow = FALSE 
      ),
      list( 
        x = 0.4,  
        y = 0.45,  
        text = paste0(x_axis_lab, "", parameter_values[6], " Fit"),
        xref = "paper",  
        yref = "paper",  
        xanchor = "center",  
        yanchor = "bottom",  
        showarrow = FALSE 
      ),  
      list( 
        x = 0.65,  
        y = 0.45,
        text = paste0(x_axis_lab, "", parameter_values[7], " Fit"),
        xref = "paper",  
        yref = "paper",  
        xanchor = "center",  
        yanchor = "bottom",  
        showarrow = FALSE 
      ),
      list( 
        x = 0.85,  
        y = 0.45,  
        text = paste0(x_axis_lab, "", parameter_values[8], " Fit"),
        xref = "paper",  
        yref = "paper",  
        xanchor = "center",  
        yanchor = "bottom",  
        showarrow = FALSE 
      )))
  
  return(allplots3)
  
}



dataset <- 3


#  1500 = 500 TRAINING + 1000 TEST, CHANGE THIS IF OPTIONS CHANGE LATER
set.seed(3)
# x1 <- runif(num_total_points, -5, 10)
# x2 <- runif(num_total_points, 0, 1)
x3 <- runif(num_total_points, -5, 5)
  
# fx1 <- ifelse(0.1 * (x1^2) < 3, 0.1 * (x1^2), 3)
# fx2 <- sin(12 * (x2 + 0.2)) / (x2 + 0.2)
fx3 <- a + (b * x3^2) + (c * x3^3)
  
set.seed(3); noise <- matrix(rnorm(num_total_points * (num_rep_sets_reg + 1), mean = 0, sd = 1), 
                  nrow = num_total_points, ncol = num_rep_sets_reg + 1)


# par(mfcol = c(1, 2))
# plot(x1, fx1); plot(x1, y1[,51])
# plot(x2, fx2); plot(x2, y2[,51])
# plot(x3, fx3); plot(x3, y3[,51])
# par(mfcol = c(1,1))
# e_test <- rnorm(num_test_points_reg, mean = 0, sd = noise_sd)
# y_test <- fx[501:1500] + e_test    # say, original y


size <- c(100, 300, 500)
error_noise <- c(1, 5, 10)
size_counter <- 3; error_noise_counter <- 3


# for(size_counter in seq_along(size))
# {
#   for(error_noise_counter in seq_along(error_noise))
#   {
    # y1 <- fx1 + noise*error_noise[error_noise_counter]
    # y2 <- fx2 + noise*error_noise[error_noise_counter]
    y3 <- fx3 + noise*error_noise[error_noise_counter]

    # change x1, x2, x3 and y1, y2, y3 as per dataset
    df_training_full <- data.frame(x = x3[1:num_training_points], fx = fx3[1:num_training_points], y_orig = y3[1:num_training_points,1])
    df_test <- data.frame(x = x3[(num_training_points+1):num_total_points], fx = fx3[(num_training_points+1):num_total_points], y_orig = y3[(num_training_points+1):num_total_points,51])
    # df_training_xs <- df_training$x; df_training_ys <- y_train
    
    bvt_calc_polynomial <- calculate_bias_variance(model_type = "polynomial",
                                                   training_data = df_training_full,
                                                   training_data_size = size[size_counter],
                                                   test_data = df_test,
                                                   parameter_values = 1:8,
                                                   fit_function = fit_polynomial,
                                                   noise_sd = error_noise[error_noise_counter])
    
    bvt_calc_spline <- calculate_bias_variance(model_type = "spline",
                                               training_data = df_training_full,
                                               training_data_size = size[size_counter],
                                               test_data = df_test,
                                               parameter_values = seq(2, 100, 13),
                                               fit_function = fit_spline,
                                               noise_sd = error_noise[error_noise_counter])
    
    bvt_calc_knn_reg <- calculate_bias_variance(model_type = "knn_reg",
                                                training_data = df_training_full,
                                                training_data_size = size[size_counter],
                                                test_data = df_test,
                                                parameter_values = seq(1, 15, 2),
                                                fit_function = fit_knn,
                                                noise_sd = error_noise[error_noise_counter])
    
    bvt_calc_tree_reg <- calculate_bias_variance(model_type = "tree_reg",
                                                 training_data = df_training_full,
                                                 training_data_size = size[size_counter],
                                                 test_data = df_test,
                                                 parameter_values = seq(2, 16, 2),
                                                 fit_function = fit_tree,
                                                 noise_sd = error_noise[error_noise_counter])
    
    bvt_calc_lasso <- calculate_bias_variance(model_type = "lasso",
                                              training_data = df_training_full,
                                              training_data_size = size[size_counter],
                                              test_data = df_test,
                                              parameter_values = seq(-4,3,1),
                                              fit_function = fit_lasso,
                                              noise_sd = error_noise[error_noise_counter])
    
    sum(bvt_calc_polynomial[[5]]==bvt_calc_spline[[5]]); sum(bvt_calc_polynomial[[5]]==bvt_calc_knn_reg[[5]]); sum(bvt_calc_polynomial[[5]]==bvt_calc_tree_reg[[5]]); sum(bvt_calc_polynomial[[5]]==bvt_calc_lasso[[5]])
    sum(bvt_calc_polynomial[[6]]==bvt_calc_spline[[6]]); sum(bvt_calc_polynomial[[6]]==bvt_calc_knn_reg[[6]]); sum(bvt_calc_polynomial[[6]]==bvt_calc_tree_reg[[6]]); sum(bvt_calc_polynomial[[6]]==bvt_calc_lasso[[6]])
    sum(bvt_calc_polynomial[[7]]==bvt_calc_spline[[7]]); sum(bvt_calc_polynomial[[7]]==bvt_calc_knn_reg[[7]]); sum(bvt_calc_polynomial[[7]]==bvt_calc_tree_reg[[7]]); sum(bvt_calc_polynomial[[7]]==bvt_calc_lasso[[7]])
    
    setwd(sprintf("C:/Users/chakraba/OneDrive - Lawrence University/Research/BVT app/app/regression-latest/data/BVTappRegression/d%s_n%s_e%s", 
                  dataset, size[size_counter], error_noise[error_noise_counter]))
    getwd()
    write.csv(bvt_calc_polynomial[[5]], "df_training_full.csv", row.names = FALSE)
    write.csv(bvt_calc_polynomial[[6]], "df_test.csv", row.names = FALSE)
    write.csv(bvt_calc_polynomial[[7]], "df_training_ys.csv", row.names = FALSE)
    write.csv(bvt_calc_polynomial[[1]], "results_polynomial.csv", row.names = FALSE)
    write.csv(bvt_calc_spline[[1]], "results_spline.csv", row.names = FALSE)
    write.csv(bvt_calc_knn_reg[[1]], "results_knn_reg.csv", row.names = FALSE)
    write.csv(bvt_calc_tree_reg[[1]], "results_tree_reg.csv", row.names = FALSE)
    write.csv(bvt_calc_lasso[[1]], "results_lasso.csv", row.names = FALSE)
    write_parquet(bvt_calc_polynomial[[8]], "df3_list_polynomial.parquet")
    write_parquet(bvt_calc_spline[[8]], "df3_list_spline.parquet")
    write_parquet(bvt_calc_knn_reg[[8]], "df3_list_knn_reg.parquet")
    write_parquet(bvt_calc_tree_reg[[8]], "df3_list_tree_reg.parquet")
    write_parquet(bvt_calc_lasso[[8]], "df3_list_lasso.parquet")
    
    rm(list = ls())
    
    
    
    
    # allinfo <- list(df_training_full = df_training_full, 
    #                      results_polynomial = bvt_calc_polynomial[[1]],
    #                      # allplots2_polynomial = second_row_plots(bvt_calc = bvt_calc_polynomial, x_axis_lab = "Degree"),
    #                      # allplots3_polynomial = third_row_plots(bvt_calc = bvt_calc_polynomial, x_axis_lab = "Degree"),
    #                      results_spline = bvt_calc_spline[[1]],
    #                      # allplots2_spline = second_row_plots(bvt_calc = bvt_calc_spline, x_axis_lab = "Degrees of Freedom"),
    #                      # allplots3_spline = third_row_plots(bvt_calc = bvt_calc_spline, x_axis_lab = "Degrees of Freedom"),
    #                      results_knn_reg = bvt_calc_knn_reg[[1]],
    #                      # allplots2_knn_reg = second_row_plots(bvt_calc = bvt_calc_knn_reg, x_axis_lab = "K"),
    #                      # allplots3_knn_reg = third_row_plots(bvt_calc = bvt_calc_knn_reg, x_axis_lab = "K"),
    #                      results_tree_reg = bvt_calc_tree_reg[[1]],
    #                      # allplots2_tree_reg = second_row_plots(bvt_calc = bvt_calc_tree_reg, x_axis_lab = "Tree Depth"),
    #                      # allplots3_tree_reg = third_row_plots(bvt_calc = bvt_calc_tree_reg, x_axis_lab = "Tree Depth"),
    #                      results_lasso = bvt_calc_lasso[[1]])
    #                      # allplots2_lasso = second_row_plots(bvt_calc = bvt_calc_lasso, x_axis_lab = "log10(\u03bb)"),
    #                      # allplots3_lasso = third_row_plots(bvt_calc = bvt_calc_lasso, x_axis_lab = "log10(\u03bb)"))
    
    
    # assign(sprintf("d3_n%s_e%s", size[size_counter], error_noise[error_noise_counter]), allinfo)
    
  
    
#   }
# }








##################################################################################################

# rm(list = setdiff(ls(), "d1_n300_e0.3"))

cb_pallete <- colorblind_pal()(8)
df_training <- d3_n300_e5$df_training_full[1:300, ]
ggplotly(ggplot(data = df_training, aes(x=x, y=y_orig)) +
  geom_point(alpha = 0.4, size = 1) +
  stat_smooth(method = "lm", se = FALSE,
              formula = y ~ poly(x, 1, raw = TRUE), color = cb_pallete[7], linewidth = 0.75) +
  stat_smooth(method = "lm", se = FALSE,
              formula = y ~ poly(x, 2, raw = TRUE), color = cb_pallete[5], linewidth = 0.75) +
  stat_smooth(method = "lm", se = FALSE,
              formula = y ~ poly(x, 3, raw = TRUE), color = cb_pallete[3], linewidth = 0.75) +
  theme_bw() +
  labs(x = "Predictor x", y = "Response y", title = "Training Data"))

d3_n300_e5$allplots2_tree_reg
d3_n300_e5$allplots3_tree_reg


saveRDS(d3_n100_e1, "d3_n100_e1.rds")
saveRDS(d3_n100_e5, "d3_n100_e5.rds")
saveRDS(d3_n100_e10, "d3_n100_e10.rds")
saveRDS(d3_n300_e1, "d3_n300_e1.rds")
saveRDS(d3_n300_e5, "d3_n300_e5.rds")
saveRDS(d3_n300_e10, "d3_n300_e10.rds")
saveRDS(d3_n500_e1, "d3_n500_e1.rds")
saveRDS(d3_n500_e5, "d3_n500_e5.rds")
saveRDS(d3_n500_e10, "d3_n500_e10.rds")

##################################################################################################

# change working directory to data
system.time(df <- readRDS("d3_n300_e1.rds"))
df$df_training_full <- df$df_training_full[1:300,]
saveRDS(df, "d3_n300_e1.rds")

##################################################################################################

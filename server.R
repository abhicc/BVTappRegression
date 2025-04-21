###########################
# old data generation and bvt calculation codes/functions, replaced by static datasets to be loaded

# # a <- 3; b <- 0.87; c <- 0.5
# num_test_points_reg <- 1000      # number of test data points
# num_rep_sets_reg <- 50                 # number of replicated datasets
# # sample_reps <- sample(1:num_rep_sets_reg, 20)
# 
# dataset <- function(){
#   
#   a <- 3; b <- 0.87; c <- 0.5
#   
#   set.seed(3)
#   x1 <- runif(1500, -5, 10)     #  1500 = 500 TRAINING + 1000 TEST, CHANGE THIS IF OPTIONS CHANGE LATER
#   x2 <- runif(1500, 0, 1)
#   x3 <- runif(1500, -5, 5)
#   
#   fx1 <- ifelse(0.1 * (x1^2) < 3, 0.1 * (x1^2), 3)
#   fx2 <- sin(12 * (x2 + 0.2)) / (x2 + 0.2)
#   fx3 <- a + (b * x3^2) + (c * x3^3)
#   
#   return(dataset = data.frame(x1, fx1, x2, fx2, x3, fx3))
# }
# 
# dataset_m <- memoise(dataset, cache = getShinyOption("cache"))
# 
# alldata <- dataset_m()
# 
# generate_data_regression <- function(input_dataset, input_num_training_points, input_noise_sd) {
#   # num_training_points <- input$num_points      # number of training data points
#   num_training_points <- input_num_training_points
#   # num_total_points <- num_training_points + num_test_points_reg
#   
#   # if(input_dataset == "Dataset 1")
#   # {
#   #   x <- runif(n = num_total_points, min = 0, max = 20)   # input/predictor
#   #   fx <- a + (b * sqrt(x)) + (c * sin(x))   # true fx
#   # }
#   # else if(input_dataset == "Dataset 2")
#   # {
#   #   x <- runif(n = num_total_points, min = -1, max = 1)   # input/predictor
#   #   fx <- sin(pi*x) # true fx
#   # }
#   # else if(input_dataset == "Dataset 3")
#   # {
#   #   x <- runif(n = num_total_points, min = -5, max = 10)   # input/predictor
#   #   fx <- ifelse(0.1*(x^2)<3, 0.1*(x^2), 3)   # true fx
#   # }
#   # else if(input_dataset == "Dataset 4")
#   # {
#   #   x <- runif(n = num_total_points, min = 0, max = 1)   # input/predictor
#   #   fx <- sin(12*(x+0.2))/(x+0.2)   # true fx
#   # }
#   # else if(input_dataset == "Dataset 5")
#   # {
#   #   x <- runif(n = num_total_points, min = -5, max = 5)   # input/predictor
#   #   fx <- a + (b * x^2) + (c * x^3)   # true fx
#   # }
#   # # else if(input$dataset == "Dataset 6")
#   # # {
#   # #   x <- runif(n = num_total_points, min = 0, max = 20)   # input/predictor
#   # #   fx <- function(x) a + (b * x) + (c * x^2)   # true fx
#   # # }
#   
#   # x <- switch(input_dataset,
#   #             # "Dataset 1" = runif(num_total_points, 0, 20),
#   #             # "Dataset 2" = runif(num_total_points, -1, 1),
#   #             "Dataset 1" = runif(num_total_points, -5, 10),
#   #             "Dataset 2" = runif(num_total_points, 0, 1),
#   #             "Dataset 3" = runif(num_total_points, -5, 5))
#   # 
#   # fx <- switch(input_dataset,
#   #              # "Dataset 1" = a + (b * sqrt(x)) + (c * sin(x)),
#   #              # "Dataset 2" = sin(pi * x),
#   #              "Dataset 1" = ifelse(0.1 * (x^2) < 3, 0.1 * (x^2), 3),
#   #              "Dataset 2" = sin(12 * (x + 0.2)) / (x + 0.2),
#   #              "Dataset 3" = a + (b * x^2) + (c * x^3))
#   
#   # x_train <- x[1:num_training_points]; fx_train <- fx[1:num_training_points]
#   # x_test <- x[-(1:num_training_points)]; fx_test <- fx[-(1:num_training_points)]
#   
#   # noise_sd <- input$epsilon          # standard deviation of epsilon
#   
#   x <- switch(input_dataset,
#               # "Dataset 1" = runif(num_total_points, 0, 20),
#               # "Dataset 2" = runif(num_total_points, -1, 1),
#               "Dataset 1" = alldata$x1,
#               "Dataset 2" = alldata$x2,
#               "Dataset 3" = alldata$x3)
#   
#   fx <- switch(input_dataset,
#                # "Dataset 1" = a + (b * sqrt(x)) + (c * sin(x)),
#                # "Dataset 2" = sin(pi * x),
#                "Dataset 1" = alldata$fx1,
#                "Dataset 2" = alldata$fx2,
#                "Dataset 3" = alldata$fx3)
#   
#   if(input_dataset == "Dataset 3")
#   {
#     noise_sd <- case_when(input_noise_sd == "Low" ~ 1, 
#                           input_noise_sd == "Medium" ~ 5,
#                           input_noise_sd == "High" ~ 10)
#   }
#   else
#   {
#     noise_sd <- case_when(input_noise_sd == "Low" ~ 0.1, 
#                           input_noise_sd == "Medium" ~ 0.5,
#                           input_noise_sd == "High" ~ 1)
#   }
#   
#   e_train <- matrix(rnorm(num_training_points * num_rep_sets_reg, mean = 0, sd = noise_sd), 
#                     nrow = num_training_points, ncol = num_rep_sets_reg)
#   y_train <- fx[1:num_training_points] + e_train
#   
#   e_test <- rnorm(num_test_points_reg, mean = 0, sd = noise_sd)
#   y_test <- fx[501:1500] + e_test    # say, original y
#   
#   df_training <- data.frame(x = x[1:num_training_points], fx = fx[1:num_training_points], y_orig = y_train[,1])
#   df_test <- data.frame(x = x[501:1500], fx = fx[501:1500], y_orig = y_test)
#   df_training_xs <- df_training$x; df_training_ys <- y_train
#   
#   # rm(e_train, y_train, x_train, fx_train, e_test, x_test, fx_test, y_test)
#   rm(e_train, y_train, e_test, y_test)
#   
#   return(list(df_training, df_test, df_training_xs, df_training_ys, noise_sd))
#   
# }
# 
# generate_data_regression_m <- memoise(generate_data_regression, cache = getShinyOption("cache"))
# 
# calculate_bias_variance <- function(model_type, df_data, parameter_values, fit_function) {
#   df_test <- df_data[[2]]
#   df_training_xs <- df_data[[3]]
#   df_training_ys <- df_data[[4]]
#   noise_sd <- df_data[[5]]
#   
#   num_training_points <- length(df_training_xs)
#   df_test_preds <- matrix(nrow = num_test_points_reg, ncol = num_rep_sets_reg)
#   train_mse <- numeric(num_rep_sets_reg)
#   
#   df_test_preds_list <- list()
#   bias_sq <- numeric(length(parameter_values))
#   variance <- numeric(length(parameter_values))
#   training_mse <- numeric(length(parameter_values))
#   
#   for (count in seq_along(parameter_values)) {
# 
#     param <- parameter_values[count]
#     
#     for (j in seq_len(num_rep_sets_reg)) {
#       y_rep <- df_training_ys[, j]
#       x_rep <- df_training_xs
#       
#       # Fit the model using the provided fit_function
#       model <- fit_function(x_rep, y_rep, param, df_test)
#       # y_hat_training <- predict(model, x_rep)
#       # y_hat_test <- predict(model, df_test$x)
#       y_hat_training <- model$y_hat_training
#       y_hat_test <- model$y_hat_test
#       
#       df_test_preds[, j] <- y_hat_test
#       train_mse[j] <- mean((y_rep - y_hat_training)^2)
#     }
#     
#     # Bias, variance, and MSE calculations
#     E_y_hat <- rowMeans(df_test_preds)
#     V_y_hat <- Rfast::rowVars(df_test_preds)
#     bias_squared <- (E_y_hat - df_test$fx)^2
#     
#     training_mse[count] <- mean(train_mse)
#     bias_sq[count] <- mean(bias_squared)
#     variance[count] <- mean(V_y_hat)
#     df_test_preds_list[[count]] <- df_test_preds
#   }
#   
#   results <- data.frame(
#     parameter_values = factor(parameter_values),
#     Bias_sq = bias_sq,
#     Variance = variance,
#     Test = bias_sq + variance + (noise_sd^2),
#     Training = training_mse
#   )
#   
#   return(list(results, parameter_values, df_test_preds_list, noise_sd))
# }
# 
# calculate_bias_variance_m <- memoise(calculate_bias_variance, cache = getShinyOption("cache"))
# 
# fit_polynomial <- function(x, y, degree, df_test) {
#   m <- lm(y ~ poly(x, degree = degree, raw = TRUE))
#   return(list(y_hat_training = m$fitted.values, y_hat_test = predict(m, newdata = data.frame(x = df_test$x))))
# }
# 
# fit_polynomial_m <- memoise(fit_polynomial, cache = getShinyOption("cache"))
# 
# fit_spline <- function(x, y, df, df_test) {
#   m <- smooth.spline(x = x, y = y, df = df)
#   return(list(y_hat_training = predict(m, x = x)$y, y_hat_test = predict(m, x = df_test$x)$y))
# }
# 
# fit_spline_m <- memoise(fit_spline, cache = getShinyOption("cache"))
# 
# fit_knn <- function(x, y, k, df_test) {
#   # list(
#   #   predict = function(new_x) {
#   #     Rfast::knn(xnew = matrix(new_x), y = y, x = matrix(x), k = k, type = "R")
#   #   }
#   # )
#   return(list(y_hat_training = Rfast::knn(xnew = matrix(x), y = y, 
#                                           x = matrix(x), k = k, type = "R"), 
#               y_hat_test = Rfast::knn(xnew = matrix(df_test$x), y = y, 
#                                       x = matrix(x), k = k, type = "R")))
# }
# 
# fit_knn_m <- memoise(fit_knn, cache = getShinyOption("cache"))
# 
# fit_tree <- function(x, y, depth, df_test) {
#   m <- rpart(y ~ x, method = "anova", control = rpart.control(cp = 0, xval = 0, minbucket = 1, maxdepth = depth))
#   return(list(y_hat_training = predict(m, newdata = data.frame(x)), y_hat_test = predict(m, newdata = data.frame(x = df_test$x))))
# }
# 
# fit_tree_m <- memoise(fit_tree, cache = getShinyOption("cache"))
# 
# fit_lasso <- function(x, y, lambda, df_test) {
#   poly_degree <- 5
#   x_rep_mat <- model.matrix(~ poly(x, degree = poly_degree, raw = TRUE))[,-1]
#   m <- glmnet(x = x_rep_mat, y = y, family = "gaussian", alpha = 0, lambda = 10^lambda)
#   return(list(y_hat_training = predict(m, newx = x_rep_mat), 
#               y_hat_test = predict(m, newx = model.matrix(~ poly(df_test$x, degree = poly_degree, raw = TRUE))[,-1])))
# }
# 
# fit_lasso_m <- memoise(fit_lasso, cache = getShinyOption("cache"))
# 
second_row_plots <- function(df_data, x_axis_lab) {
  # results <- df_data[[1]]
  # noise_sd <- df_data[[4]]
  results <- df_data
  noise_sd <- 0.1

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

  mse_plot <- plot_ly(data = results, x = ~factor(parameter_values), y = ~Test,
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

  bias_test_var_plot <- plot_ly(data = results, x = ~factor(parameter_values), y = ~Test,
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

  bias_plot <- plot_ly(data = results, x = ~factor(parameter_values), y = ~Bias_sq,
                       name = 'Squared Bias', type = "bar", color = I("#CC79A7")) %>%
    # add_trace(y = ~Variance, name = 'Variance', type = "bar", color = I("#009E73")) %>%
    # add_trace(y = ~Variance, name = 'Variance', mode = 'lines+markers', color = I("#009E73")) %>%
    layout(xaxis = list(title = x_axis_lab, showline = TRUE, zeroline = FALSE),
           yaxis = list(title = "Estimated Squared Bias", showline = TRUE, zeroline = FALSE))


  # variance_plot <- ggplot(data = results) +
  #   geom_col(aes(x = parameter_values, y = Variance), fill = cb_pallete[4]) + theme_bw() +
  #   labs(y = "Estimated Variance", x = x_axis_lab)

  variance_plot <- plot_ly(data = results, x = ~factor(parameter_values), y = ~Variance,
                           name = 'Variance', type = "bar", color = I("#009E73")) %>%
    layout(xaxis = list(title = x_axis_lab, showline = TRUE, zeroline = FALSE),
           yaxis = list(title = "Estimated Variance", showline = TRUE, zeroline = FALSE))

  allplots2 <- subplot(mse_plot, bias_test_var_plot, bias_plot, variance_plot, nrows = 1, shareX = TRUE)

  return(allplots2)
}
# 
# second_row_plots_m <- memoise(second_row_plots, cache = getShinyOption("cache"))
# 
third_row_plots <- function(model_type, parameter_values, df_data2, x_axis_lab) {
  # df_test <- df_data1[[2]]
  # parameter_values <- df_data2[[2]]
  # df_test_preds_list <- df_data2[[3]]
  parameter_values <- parameter_values
  # df3 <- df_data2

  # min_y_limit <- min(sapply(df_data2, function(e) min(e[,5])))
  # max_y_limit <- max(sapply(df_data2, function(e) max(e[,5])))
  min_y_limit <- min(df_data2$preds)
  max_y_limit <- max(df_data2$preds)

  # min_y_limit <- min(sapply(df_test_preds_list, min), min(df_test$y_orig))
  # max_y_limit <- max(sapply(df_test_preds_list, max), max(df_test$y_orig))


  replicated_datasets_graphs <- list()
  for(count in seq_along(parameter_values))
  {
    # df_test_preds_selected <- data.frame(df_test_preds_list[[count]][,sample_reps])
    # df_test_preds_selected <- data.frame(df_test_preds_list[[count]])
    # df_test_preds_selected$mean_pred <- rowMeans(df_test_preds_list[[count]])
    # df_test_preds_selected$expl <- df_test$x
    # df_test_preds_selected_pivot <- df_test_preds_selected %>%
    #   pivot_longer(cols = starts_with("X"), names_to = "rep_data", values_to = "preds")

    replicated_datasets_graphs[[count]] <- plot_ly(data = df_data2 %>%
                                                     filter(parameter_val == parameter_values[count]) %>%
                                                     arrange(expl),
                                                   x = ~expl, y = ~fx,
                                                   name = 'true f(x)', type = 'scatter', mode = 'lines', color = I("#000000"), line = list(width = 3)) %>%
      add_trace(x = ~expl, y = ~preds, name = 'replicated fit', mode = 'lines', color = I("orange"), line = list(dash = "dot", width = 0.5)) %>%
      add_trace(x = ~expl, y = ~mean_pred, name = 'average fit', mode = 'lines', color = I("#56B4E9"), line = list(width = 2)) %>%
      layout(xaxis = list(title = "x", showline = TRUE, zeroline = FALSE),
             yaxis = list(title = "", showline = TRUE, zeroline = FALSE),
             legend = list(orientation = 'h', x = 0, y = 1.1))


    # plot_ly(data = df_test %>% arrange(x), x = ~x, y = ~fx,
    #                                                name = 'true f(x)', type = 'scatter', mode = 'lines', color = I("#000000"), line = list(width = 3)) %>%
    #   add_trace(data = df_test_preds_selected_pivot %>% arrange(expl), x = ~expl, y = ~preds, group_by = ~rep_data,
    #             name = 'replicated fit', mode = 'lines', color = I("orange"), line = list(dash = "dot", width = 0.5)) %>%
    #   add_trace(data = df_test_preds_selected %>% arrange(expl), x = ~expl, y = ~mean_pred,
    #             name = 'average fit', mode = 'lines', color = I("#56B4E9"), line = list(width = 2)) %>%
    #   layout(xaxis = list(title = "x", showline = TRUE, zeroline = FALSE),
    #          yaxis = list(title = "", showline = TRUE, zeroline = FALSE),
    #          legend = list(orientation = 'h', x = 0, y = 1.1))

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
        text = paste0(x_axis_lab, " ", parameter_values[1], " Fit"),
        xref = "paper",
        yref = "paper",
        xanchor = "center",
        yanchor = "bottom",
        showarrow = FALSE
      ),
      list(
        x = 0.4,
        y = 1,
        text = paste0(x_axis_lab, " ", parameter_values[2], " Fit"),
        xref = "paper",
        yref = "paper",
        xanchor = "center",
        yanchor = "bottom",
        showarrow = FALSE
      ),
      list(
        x = 0.65,
        y = 1,
        text = paste0(x_axis_lab, " ", parameter_values[3], " Fit"),
        xref = "paper",
        yref = "paper",
        xanchor = "center",
        yanchor = "bottom",
        showarrow = FALSE
      ),
      list(
        x = 0.85,
        y = 1,
        text = paste0(x_axis_lab, " ", parameter_values[4], " Fit"),
        xref = "paper",
        yref = "paper",
        xanchor = "center",
        yanchor = "bottom",
        showarrow = FALSE
      ),
      list(
        x = 0.1,
        y = 0.45,
        text = paste0(x_axis_lab, " ", parameter_values[5], " Fit"),
        xref = "paper",
        yref = "paper",
        xanchor = "center",
        yanchor = "bottom",
        showarrow = FALSE
      ),
      list(
        x = 0.4,
        y = 0.45,
        text = paste0(x_axis_lab, " ", parameter_values[6], " Fit"),
        xref = "paper",
        yref = "paper",
        xanchor = "center",
        yanchor = "bottom",
        showarrow = FALSE
      ),
      list(
        x = 0.65,
        y = 0.45,
        text = paste0(x_axis_lab, " ", parameter_values[7], " Fit"),
        xref = "paper",
        yref = "paper",
        xanchor = "center",
        yanchor = "bottom",
        showarrow = FALSE
      ),
      list(
        x = 0.85,
        y = 0.45,
        text = paste0(x_axis_lab, " ", parameter_values[8], " Fit"),
        xref = "paper",
        yref = "paper",
        xanchor = "center",
        yanchor = "bottom",
        showarrow = FALSE
      )))

  return(allplots3)

}
# 
# third_row_plots_m <- memoise(third_row_plots, cache = getShinyOption("cache"))
# 

###########################

function(input, output, session) {
  

  ####################
  # generate data for regression and link different inputs
  
  df_data_regression <- eventReactive(input$action_polynomial | input$action_spline | input$action_knn_reg | input$action_tree_reg | input$action_lasso | input$action_rf, 
    
  {
    
    dataset_id <- switch(input$dataset_polynomial,
                         "Dataset 1" = 1,
                         "Dataset 2" = 2,
                         "Dataset 3" = 3)
    
    size_id <- as.numeric(input$num_points_polynomial)
    
    if(input$dataset_polynomial == "Dataset 3")
    {
      noise_id <- case_when(input$epsilon_polynomial == "Low" ~ 1,
                            input$epsilon_polynomial == "Medium" ~ 5,
                            input$epsilon_polynomial == "High" ~ 10)
    }
    else
    {
      noise_id <- case_when(input$epsilon_polynomial == "Low" ~ 0.1,
                            input$epsilon_polynomial == "Medium" ~ 0.5,
                            input$epsilon_polynomial == "High" ~ 1)
    }  


    # df_training <- fread(sprintf("https://raw.githubusercontent.com/abhicc/BVTappRegression/refs/heads/main/d%s_n%s_e%s/df_training_full.csv", dataset_id, size_id, noise_id))
    # df_test <- fread(sprintf("https://raw.githubusercontent.com/abhicc/BVTappRegression/refs/heads/main/d%s_n%s_e%s/df_test.csv", dataset_id, size_id, noise_id))
    # results_polynomial <- fread(sprintf("https://raw.githubusercontent.com/abhicc/BVTappRegression/refs/heads/main/d%s_n%s_e%s/results_polynomial.csv", dataset_id, size_id, noise_id))
    # results_spline <- fread(sprintf("https://raw.githubusercontent.com/abhicc/BVTappRegression/refs/heads/main/d%s_n%s_e%s/results_spline.csv", dataset_id, size_id, noise_id))
    # results_knn_reg <- fread(sprintf("https://raw.githubusercontent.com/abhicc/BVTappRegression/refs/heads/main/d%s_n%s_e%s/results_knn_reg.csv", dataset_id, size_id, noise_id))
    # results_tree_reg <- fread(sprintf("https://raw.githubusercontent.com/abhicc/BVTappRegression/refs/heads/main/d%s_n%s_e%s/results_tree_reg.csv", dataset_id, size_id, noise_id))
    # results_lasso <- fread(sprintf("https://raw.githubusercontent.com/abhicc/BVTappRegression/refs/heads/main/d%s_n%s_e%s/results_lasso.csv", dataset_id, size_id, noise_id))
    # df3_list_polynomial <- fread(sprintf("https://raw.githubusercontent.com/abhicc/BVTappRegression/refs/heads/main/d%s_n%s_e%s/df3_list_polynomial.csv", dataset_id, size_id, noise_id))
    
    df_training <- fread(sprintf("BVTappRegression/d%s_n%s_e%s/df_training_full.csv", dataset_id, size_id, noise_id))
    df_training_ys <- fread(sprintf("BVTappRegression/d%s_n%s_e%s/df_training_ys.csv", dataset_id, size_id, noise_id))
    df_test <- fread(sprintf("BVTappRegression/d%s_n%s_e%s/df_test.csv", dataset_id, size_id, noise_id))
    results_polynomial <- fread(sprintf("BVTappRegression/d%s_n%s_e%s/results_polynomial.csv", dataset_id, size_id, noise_id))
    results_spline <- fread(sprintf("BVTappRegression/d%s_n%s_e%s/results_spline.csv", dataset_id, size_id, noise_id))
    results_knn_reg <- fread(sprintf("BVTappRegression/d%s_n%s_e%s/results_knn_reg.csv", dataset_id, size_id, noise_id))
    results_tree_reg <- fread(sprintf("BVTappRegression/d%s_n%s_e%s/results_tree_reg.csv", dataset_id, size_id, noise_id))
    results_lasso <- fread(sprintf("BVTappRegression/d%s_n%s_e%s/results_lasso.csv", dataset_id, size_id, noise_id))
    # df3_list_polynomial <- fread(sprintf("data/BVTappRegression/d%s_n%s_e%s/df3_list_polynomial.csv", dataset_id, size_id, noise_id))
    df3_list_polynomial <- read_parquet(sprintf("BVTappRegression/d%s_n%s_e%s/df3_list_polynomial.parquet", dataset_id, size_id, noise_id))
    df3_list_spline <- read_parquet(sprintf("BVTappRegression/d%s_n%s_e%s/df3_list_spline.parquet", dataset_id, size_id, noise_id))
    df3_list_knn_reg <- read_parquet(sprintf("BVTappRegression/d%s_n%s_e%s/df3_list_knn_reg.parquet", dataset_id, size_id, noise_id))
    df3_list_tree_reg <- read_parquet(sprintf("BVTappRegression/d%s_n%s_e%s/df3_list_tree_reg.parquet", dataset_id, size_id, noise_id))
    df3_list_lasso <- read_parquet(sprintf("BVTappRegression/d%s_n%s_e%s/df3_list_lasso.parquet", dataset_id, size_id, noise_id))
    # df_test_preds_selected_pivot_polynomial <- readRDS(sprintf("data/d%s_n%s_e%s/df_test_preds_selected_pivot_polynomial.rds", dataset_id, size_id, noise_id))
    # df_test_preds_selected_pivot_spline <- readRDS(sprintf("data/d%s_n%s_e%s/df_test_preds_selected_pivot_spline.rds", dataset_id, size_id, noise_id))
    # df_test_preds_selected_pivot_knn_reg <- readRDS(sprintf("data/d%s_n%s_e%s/df_test_preds_selected_pivot_knn_reg.rds", dataset_id, size_id, noise_id))
    # df_test_preds_selected_pivot_lasso <- readRDS(sprintf("data/d%s_n%s_e%s/df_test_preds_selected_pivot_tree_reg.rds", dataset_id, size_id, noise_id))
    # df_test_preds_selected_pivot_tree_reg <- readRDS(sprintf("data/d%s_n%s_e%s/df_test_preds_selected_pivot_lasso.rds", dataset_id, size_id, noise_id))
    # p1 <- readRDS(sprintf("data/d%s_n%s_e%s/p1.rds", dataset_id, size_id, noise_id))
    # p2 <- readRDS(sprintf("data/d%s_n%s_e%s/p2.rds", dataset_id, size_id, noise_id))
    # p3 <- readRDS(sprintf("data/d%s_n%s_e%s/p3.rds", dataset_id, size_id, noise_id))
    # p4 <- readRDS(sprintf("data/d%s_n%s_e%s/p4.rds", dataset_id, size_id, noise_id))
    # p5 <- readRDS(sprintf("data/d%s_n%s_e%s/p5.rds", dataset_id, size_id, noise_id))
    # p6 <- readRDS(sprintf("data/d%s_n%s_e%s/p6.rds", dataset_id, size_id, noise_id))
    # p7 <- readRDS(sprintf("data/d%s_n%s_e%s/p7.rds", dataset_id, size_id, noise_id))
    # p8 <- readRDS(sprintf("data/d%s_n%s_e%s/p8.rds", dataset_id, size_id, noise_id))
    # 
    
    
    # dflist <- readRDS(sprintf("data/d%s_n%s_e%s.rds", dataset_id, size_id, noise_id))
      
    # dflist <- generate_data_regression_m(input_dataset = input$dataset_polynomial,
    #                                      input_num_training_points = input$num_points_polynomial,
    #                                      input_noise_sd = input$epsilon_polynomial)
    
    # return(list(dflist, size_id))
    return(list(df_training, df_test,
                results_polynomial, results_spline, results_knn_reg, results_tree_reg, results_lasso,
                df3_list_polynomial, df3_list_spline, df3_list_knn_reg, df3_list_tree_reg, df3_list_lasso,
                df_training_ys))
    # return(list(df_training, df_test,
    #             results_polynomial))
                # p1, p2, p3, p4, p5, p6, p7, p8))
           # , 
                # df_test_preds_selected_pivot_spline, 
                # df_test_preds_selected_pivot_knn_reg, df_test_preds_selected_pivot_tree_reg,
                # df_test_preds_selected_pivot_lasso))
    
  }) 
  
  # %>% bindCache(input$action_polynomial, input$dataset_polynomial, input$num_points_polynomial, input$epsilon_polynomial)
  

  ###########################
  
  observeEvent(input$dataset_polynomial, {
    updateSelectInput(inputId = "dataset_spline", selected = input$dataset_polynomial)
    updateSelectInput(inputId = "dataset_knn_reg", selected = input$dataset_polynomial)
    updateSelectInput(inputId = "dataset_tree_reg", selected = input$dataset_polynomial)
    updateSelectInput(inputId = "dataset_lasso", selected = input$dataset_polynomial)
    updateSelectInput(inputId = "dataset_rf", selected = input$dataset_polynomial)
  })

  observeEvent(input$dataset_spline, {
    updateSelectInput(inputId = "dataset_polynomial", selected = input$dataset_spline)
    updateSelectInput(inputId = "dataset_knn_reg", selected = input$dataset_spline)
    updateSelectInput(inputId = "dataset_tree_reg", selected = input$dataset_spline)
    updateSelectInput(inputId = "dataset_lasso", selected = input$dataset_spline)
    updateSelectInput(inputId = "dataset_rf", selected = input$dataset_spline)
  })

  observeEvent(input$dataset_knn_reg, {
    updateSelectInput(inputId = "dataset_spline", selected = input$dataset_knn_reg)
    updateSelectInput(inputId = "dataset_polynomial", selected = input$dataset_knn_reg)
    updateSelectInput(inputId = "dataset_tree_reg", selected = input$dataset_knn_reg)
    updateSelectInput(inputId = "dataset_lasso", selected = input$dataset_knn_reg)
    updateSelectInput(inputId = "dataset_rf", selected = input$dataset_knn_reg)
  })

  observeEvent(input$dataset_tree_reg, {
    updateSelectInput(inputId = "dataset_spline", selected = input$dataset_tree_reg)
    updateSelectInput(inputId = "dataset_knn_reg", selected = input$dataset_tree_reg)
    updateSelectInput(inputId = "dataset_polynomial", selected = input$dataset_tree_reg)
    updateSelectInput(inputId = "dataset_lasso", selected = input$dataset_tree_reg)
    updateSelectInput(inputId = "dataset_rf", selected = input$dataset_tree_reg)
  })

  observeEvent(input$dataset_lasso, {
    updateSelectInput(inputId = "dataset_spline", selected = input$dataset_lasso)
    updateSelectInput(inputId = "dataset_knn_reg", selected = input$dataset_lasso)
    updateSelectInput(inputId = "dataset_tree_reg", selected = input$dataset_lasso)
    updateSelectInput(inputId = "dataset_polynomial", selected = input$dataset_lasso)
    updateSelectInput(inputId = "dataset_rf", selected = input$dataset_lasso)
  })

  observeEvent(input$dataset_rf, {
    updateSelectInput(inputId = "dataset_lasso", selected = input$dataset_rf)
    updateSelectInput(inputId = "dataset_spline", selected = input$dataset_rf)
    updateSelectInput(inputId = "dataset_knn_reg", selected = input$dataset_rf)
    updateSelectInput(inputId = "dataset_tree_reg", selected = input$dataset_rf)
    updateSelectInput(inputId = "dataset_polynomial", selected = input$dataset_rf)
  })



  observeEvent(input$num_points_polynomial, {
    updateSliderInput(inputId = "num_points_spline", value = input$num_points_polynomial)
    updateSliderInput(inputId = "num_points_knn_reg", value = input$num_points_polynomial)
    updateSliderInput(inputId = "num_points_tree_reg", value = input$num_points_polynomial)
    updateSliderInput(inputId = "num_points_lasso", value = input$num_points_polynomial)
    updateSliderInput(inputId = "num_points_rf", value = input$num_points_polynomial)
  })

  observeEvent(input$num_points_spline, {
    updateSliderInput(inputId = "num_points_polynomial", value = input$num_points_spline)
    updateSliderInput(inputId = "num_points_knn_reg", value = input$num_points_spline)
    updateSliderInput(inputId = "num_points_tree_reg", value = input$num_points_spline)
    updateSliderInput(inputId = "num_points_lasso", value = input$num_points_spline)
    updateSliderInput(inputId = "num_points_rf", value = input$num_points_spline)
  })

  observeEvent(input$num_points_knn_reg, {
    updateSliderInput(inputId = "num_points_spline", value = input$num_points_knn_reg)
    updateSliderInput(inputId = "num_points_polynomial", value = input$num_points_knn_reg)
    updateSliderInput(inputId = "num_points_tree_reg", value = input$num_points_knn_reg)
    updateSliderInput(inputId = "num_points_lasso", value = input$num_points_knn_reg)
    updateSliderInput(inputId = "num_points_rf", value = input$num_points_knn_reg)
  })

  observeEvent(input$num_points_tree_reg, {
    updateSliderInput(inputId = "num_points_spline", value = input$num_points_tree_reg)
    updateSliderInput(inputId = "num_points_knn_reg", value = input$num_points_tree_reg)
    updateSliderInput(inputId = "num_points_polynomial", value = input$num_points_tree_reg)
    updateSliderInput(inputId = "num_points_lasso", value = input$num_points_tree_reg)
    updateSliderInput(inputId = "num_points_rf", value = input$num_points_tree_reg)
  })

  observeEvent(input$num_points_lasso, {
    updateSliderInput(inputId = "num_points_spline", value = input$num_points_lasso)
    updateSliderInput(inputId = "num_points_knn_reg", value = input$num_points_lasso)
    updateSliderInput(inputId = "num_points_tree_reg", value = input$num_points_lasso)
    updateSliderInput(inputId = "num_points_polynomial", value = input$num_points_lasso)
    updateSliderInput(inputId = "num_points_rf", value = input$num_points_lasso)
  })

  observeEvent(input$num_points_rf, {
    updateSliderInput(inputId = "num_points_spline", value = input$num_points_rf)
    updateSliderInput(inputId = "num_points_knn_reg", value = input$num_points_rf)
    updateSliderInput(inputId = "num_points_tree_reg", value = input$num_points_rf)
    updateSliderInput(inputId = "num_points_polynomial", value = input$num_points_rf)
    updateSliderInput(inputId = "num_points_lasso", value = input$num_points_rf)
  })



  observeEvent(input$epsilon_polynomial, {
    updateSliderTextInput(session = session, inputId = "epsilon_spline", selected = input$epsilon_polynomial)
    updateSliderTextInput(session = session, inputId = "epsilon_knn_reg", selected = input$epsilon_polynomial)
    updateSliderTextInput(session = session, inputId = "epsilon_tree_reg", selected = input$epsilon_polynomial)
    updateSliderTextInput(session = session, inputId = "epsilon_lasso", selected = input$epsilon_polynomial)
    updateSliderTextInput(session = session, inputId = "epsilon_rf", selected = input$epsilon_polynomial)
  })

  observeEvent(input$epsilon_spline, {
    updateSliderTextInput(session = session, inputId = "epsilon_polynomial", selected = input$epsilon_spline)
    updateSliderTextInput(session = session, inputId = "epsilon_knn_reg", selected = input$epsilon_spline)
    updateSliderTextInput(session = session, inputId = "epsilon_tree_reg", selected = input$epsilon_spline)
    updateSliderTextInput(session = session, inputId = "epsilon_lasso", selected = input$epsilon_spline)
    updateSliderTextInput(session = session, inputId = "epsilon_rf", selected = input$epsilon_spline)
  })

  observeEvent(input$epsilon_knn_reg, {
    updateSliderTextInput(session = session, inputId = "epsilon_spline", selected = input$epsilon_knn_reg)
    updateSliderTextInput(session = session, inputId = "epsilon_polynomial", selected = input$epsilon_knn_reg)
    updateSliderTextInput(session = session, inputId = "epsilon_tree_reg", selected = input$epsilon_knn_reg)
    updateSliderTextInput(session = session, inputId = "epsilon_lasso", selected = input$epsilon_knn_reg)
    updateSliderTextInput(session = session, inputId = "epsilon_rf", selected = input$epsilon_knn_reg)
  })

  observeEvent(input$epsilon_tree_reg, {
    updateSliderTextInput(session = session, inputId = "epsilon_spline", selected = input$epsilon_tree_reg)
    updateSliderTextInput(session = session, inputId = "epsilon_knn_reg", selected = input$epsilon_tree_reg)
    updateSliderTextInput(session = session, inputId = "epsilon_polynomial", selected = input$epsilon_tree_reg)
    updateSliderTextInput(session = session, inputId = "epsilon_lasso", selected = input$epsilon_tree_reg)
    updateSliderTextInput(session = session, inputId = "epsilon_rf", selected = input$epsilon_tree_reg)
  })

  observeEvent(input$epsilon_lasso, {
    updateSliderTextInput(session = session, inputId = "epsilon_spline", selected = input$epsilon_lasso)
    updateSliderTextInput(session = session, inputId = "epsilon_knn_reg", selected = input$epsilon_lasso)
    updateSliderTextInput(session = session, inputId = "epsilon_tree_reg", selected = input$epsilon_lasso)
    updateSliderTextInput(session = session, inputId = "epsilon_polynomial", selected = input$epsilon_lasso)
    updateSliderTextInput(session = session, inputId = "epsilon_rf", selected = input$epsilon_lasso)
  })

  observeEvent(input$epsilon_rf, {
    updateSliderTextInput(session = session, inputId = "epsilon_spline", selected = input$epsilon_rf)
    updateSliderTextInput(session = session, inputId = "epsilon_knn_reg", selected = input$epsilon_rf)
    updateSliderTextInput(session = session, inputId = "epsilon_tree_reg", selected = input$epsilon_rf)
    updateSliderTextInput(session = session, inputId = "epsilon_polynomial", selected = input$epsilon_rf)
    updateSliderTextInput(session = session, inputId = "epsilon_lasso", selected = input$epsilon_rf)
  })



  # observeEvent(input$action_polynomial, {
  #   updateActionButton(session, inputId = "action_spline")
  #   updateActionButton(session, inputId = "action_knn_reg")
  #   updateActionButton(session, inputId = "action_tree_reg")
  #   updateActionButton(session, inputId = "action_lasso")
  #   updateActionButton(session, inputId = "action_rf")
  # })
  # 
  # observeEvent(input$action_spline, {
  #   updateActionButton(session, inputId = "action_polynomial")
  #   updateActionButton(session, inputId = "action_knn_reg")
  #   updateActionButton(session, inputId = "action_tree_reg")
  #   updateActionButton(session, inputId = "action_lasso")
  #   updateActionButton(session, inputId = "action_rf")
  # })
  # 
  # observeEvent(input$action_knn_reg, {
  #   updateActionButton(session, inputId = "action_spline")
  #   updateActionButton(session, inputId = "action_polynomial")
  #   updateActionButton(session, inputId = "action_tree_reg")
  #   updateActionButton(session, inputId = "action_lasso")
  #   updateActionButton(session, inputId = "action_rf")
  # })
  # 
  # observeEvent(input$action_tree_reg, {
  #   updateActionButton(session, inputId = "action_spline")
  #   updateActionButton(session, inputId = "action_knn_reg")
  #   updateActionButton(session, inputId = "action_polynomial")
  #   updateActionButton(session, inputId = "action_lasso")
  #   updateActionButton(session, inputId = "action_rf")
  # })
  # 
  # observeEvent(input$action_lasso, {
  #   updateActionButton(session, inputId = "action_spline")
  #   updateActionButton(session, inputId = "action_knn_reg")
  #   updateActionButton(session, inputId = "action_tree_reg")
  #   updateActionButton(session, inputId = "action_polynomial")
  #   updateActionButton(session, inputId = "action_rf")
  # })
  # 
  # observeEvent(input$action_rf, {
  #   updateActionButton(session, inputId = "action_spline")
  #   updateActionButton(session, inputId = "action_knn_reg")
  #   updateActionButton(session, inputId = "action_tree_reg")
  #   updateActionButton(session, inputId = "action_polynomial")
  #   updateActionButton(session, inputId = "action_lasso")
  # })
  
  ####################
  # calculate bias and variance 
  
  # df_bvt_calc_polynomial <- reactive({
  #   calculate_bias_variance_m(
  #     model_type = "polynomial",
  #     df_data = df_data_regression(),
  #     parameter_values = 1:8,
  #     fit_function = fit_polynomial_m
  #   )
  # }) %>% bindCache(df_data_regression())
  # 
  # df_bvt_calc_spline <- reactive({
  #   calculate_bias_variance_m(
  #     model_type = "spline",
  #     df_data = df_data_regression(),
  #     parameter_values = seq(2, 100, 13),
  #     fit_function = fit_spline_m
  #   )
  # }) %>% bindCache(df_data_regression())
  # 
  # df_bvt_calc_knn_reg <- reactive({
  #   calculate_bias_variance_m(
  #     model_type = "knn_reg",
  #     df_data = df_data_regression(),
  #     parameter_values = seq(1, 15, 2),
  #     fit_function = fit_knn_m
  #   )
  # }) %>% bindCache(df_data_regression())
  # 
  # df_bvt_calc_tree_reg <- reactive({
  #   calculate_bias_variance_m(
  #     model_type = "tree_reg",
  #     df_data = df_data_regression(),
  #     parameter_values = seq(2, 16, 2),
  #     fit_function = fit_tree_m
  #   )
  # }) %>% bindCache(df_data_regression())
  # 
  # df_bvt_calc_lasso <- reactive({
  #   calculate_bias_variance_m(
  #     model_type = "lasso",
  #     df_data = df_data_regression(),
  #     parameter_values = seq(-4,3,1),
  #     fit_function = fit_lasso_m
  #   )
  #   
  # }) %>% bindCache(df_data_regression())
  
  df_bvt_calc_rf <- reactive({
    
    # df_training <- df_data_regression()[[1]]
    
    df_training_subset <- df_data_regression()[[1]][1:30, ]
    df_training_subset$Var1 <- rownames(df_training_subset)
    
    x_seq <- seq(min(df_training_subset$x, na.rm = TRUE), max(df_training_subset$x, na.rm = TRUE), 0.001)
    
    rffit <- ranger(formula = y_orig ~ x, data = df_training_subset,
                    # x = data.frame(df_training_subset$x), y = df_training_subset$y_orig, xtest = data.frame(x_seq), #data = data.frame(y_rep = y_rep, x_rep = x_rep),
                    num.trees = 10, # number of trees to grow (bootstrap samples) usually 500
                    mtry = 1, 
                    min.bucket = 1,
                    keep.inbag=TRUE)
    preds_rf <- predict(rffit, data = data.frame(x = x_seq), predict.all = TRUE)                        # rffit$test$predicted
    
    # return(list(results, parameter_values, df_test_preds_selected_pivot, noise_sd))
    return(list(df_training_subset, x_seq, rffit, preds_rf))
    
  }) %>% bindCache(df_data_regression())
  
  
  ###########################
  # old bias-variance calculation code, replaced by a function and code above
  
  # df_bvt_calc_polynomial <- reactive({
  #   
  #   df_test <- df_data_regression()[[2]]
  #   df_training_xs <- df_data_regression()[[3]]
  #   df_training_ys <- df_data_regression()[[4]]
  #   noise_sd <- df_data_regression()[[5]]
  #   
  #   num_training_points <- input$num_points_polynomial
  #   # noise_sd <- input$epsilon_polynomial
  #   
  #   df_training_preds <- matrix(nrow = num_training_points, ncol = num_rep_sets_reg)
  #   df_test_preds <- matrix(nrow = num_test_points_reg, ncol = num_rep_sets_reg)
  #   train_mse <- rep(NA, num_rep_sets_reg)
  #   # df_test_preds_selected_pivot <- list()
  #   df_test_preds_list <- list()
  #   # train_mse_list <- list()
  #   
  #   parameter_values <- 1:8; l <- length(parameter_values)
  #   bias_sq <- numeric(l); variance <- numeric(l); training_mse <- numeric(l)
  #   
  #   for(count in seq_along(parameter_values))
  #   {
  #     d <- parameter_values[count]
  #     
  #     # FOR AVERAGE TRAINING AND TEST MSE FROM REPLICATED DATASETS AND TEST SET
  #     for(j in 1:num_rep_sets_reg)
  #     {
  #       
  #       # fit model of d degrees of freedom to each replicated (training) set
  #       # get predictions (y_hat's) for each replicated (training) set and test set
  #       y_rep <- df_training_ys[,j]; x_rep <- df_training_xs
  #       m <- lm(y_rep ~ poly(x_rep, degree = d, raw = TRUE))
  #       y_hat_training <- m$fitted.values
  #       y_hat_test <- predict(m, newdata = data.frame(x_rep = df_test$x))
  #       
  #       # store predictions for each set
  #       df_training_preds[,j] <- y_hat_training
  #       df_test_preds[,j] <- y_hat_test
  #       
  #       # calculate training MSE for each replicated (training) dataset
  #       train_mse[j] <- mean((y_rep-y_hat_training)^2)
  #       
  #     }
  #     
  #     df_test_preds_list[[count]] <- df_test_preds
  #     
  #     # calculate bias squared and variance for test MSE
  #     E_y_hat <- apply(df_test_preds, 1, mean)
  #     V_y_hat <- apply(df_test_preds, 1, var)
  #     bias_squared <- (E_y_hat - df_test$fx)^2
  #     
  #     # store bias squared, variance, and training MSE values
  #     training_mse[count] <- mean(train_mse)
  #     bias_sq[count] <- mean(bias_squared)
  #     variance[count] <- mean(V_y_hat)
  #     
  #     
  #   }
  #   
  #   
  #   # store results
  #   results <- data.frame(parameter_values = factor(parameter_values),
  #                         Bias_sq = bias_sq,
  #                         Variance = variance,
  #                         Test = bias_sq + variance + (noise_sd^2),
  #                         Training = training_mse)
  #   
  # 
  #   # return(list(results, parameter_values, df_test_preds_selected_pivot, noise_sd))
  #   return(list(results, parameter_values, df_test_preds_list, noise_sd))
  #   
  # })
  # 
  # df_bvt_calc_spline <- reactive({
  #   
  #   df_test <- df_data_regression()[[2]]
  #   df_training_xs <- df_data_regression()[[3]]
  #   df_training_ys <- df_data_regression()[[4]]
  #   noise_sd <- df_data_regression()[[5]]
  #   
  #   num_training_points <- input$num_points_polynomial
  #   # ; noise_sd <- input$epsilon_polynomial
  #   
  #   df_training_preds <- matrix(nrow = num_training_points, ncol = num_rep_sets_reg)
  #   df_test_preds <- matrix(nrow = num_test_points_reg, ncol = num_rep_sets_reg)
  #   train_mse <- rep(NA, num_rep_sets_reg)
  #   # df_test_preds_selected_pivot <- list()
  #   df_test_preds_list <- list()
  #   # train_mse_list <- list()
  #   
  #   parameter_values <- seq(2, 100, 13); l <- length(parameter_values)
  #   bias_sq <- numeric(l); variance <- numeric(l); training_mse <- numeric(l)
  #   
  #   for(count in seq_along(parameter_values))
  #   {
  #     d <- parameter_values[count]
  #     
  #     # FOR AVERAGE TRAINING AND TEST MSE FROM REPLICATED DATASETS AND TEST SET
  #     for(j in 1:num_rep_sets_reg)
  #     {
  #       
  #       # fit model of d degrees of freedom to each replicated (training) set
  #       # get predictions (y_hat's) for each replicated (training) set and test set
  #       y_rep <- df_training_ys[,j]; x_rep <- df_training_xs
  #       m <- smooth.spline(x = x_rep, y = y_rep, df = d)
  #       y_hat_training <- predict(m, x = x_rep)$y
  #       y_hat_test <- predict(m, x = df_test$x)$y
  #       
  #       # store predictions for each set
  #       df_training_preds[,j] <- y_hat_training
  #       df_test_preds[,j] <- y_hat_test
  #       
  #       # calculate training MSE for each replicated (training) dataset
  #       train_mse[j] <- mean((y_rep-y_hat_training)^2)
  #       
  #     }
  #     
  #     df_test_preds_list[[count]] <- df_test_preds
  # 
  #     # calculate bias squared and variance for test MSE
  #     E_y_hat <- apply(df_test_preds, 1, mean)
  #     V_y_hat <- apply(df_test_preds, 1, var)
  #     bias_squared <- (E_y_hat - df_test$fx)^2
  #     
  #     # store bias squared, variance, and training MSE values
  #     training_mse[count] <- mean(train_mse)
  #     bias_sq[count] <- mean(bias_squared)
  #     variance[count] <- mean(V_y_hat)
  #     
  #     
  #   }
  #   
  #   
  #   # store results
  #   results <- data.frame(parameter_values = factor(parameter_values),
  #                         Bias_sq = bias_sq,
  #                         Variance = variance,
  #                         Test = bias_sq + variance + (noise_sd^2),
  #                         Training = training_mse)
  #   
  #   # return(list(results, parameter_values, df_test_preds_selected_pivot, noise_sd))
  #   return(list(results, parameter_values, df_test_preds_list, noise_sd))
  #   
  # })
  # 
  # df_bvt_calc_knn_reg <- reactive({
  #   
  #   df_test <- df_data_regression()[[2]]
  #   df_training_xs <- df_data_regression()[[3]]
  #   df_training_ys <- df_data_regression()[[4]]
  #   noise_sd <- df_data_regression()[[5]]
  #   
  #   num_training_points <- input$num_points_polynomial
  #   # ; noise_sd <- input$epsilon_polynomial
  #   
  #   df_training_preds <- matrix(nrow = num_training_points, ncol = num_rep_sets_reg)
  #   df_test_preds <- matrix(nrow = num_test_points_reg, ncol = num_rep_sets_reg)
  #   train_mse <- rep(NA, num_rep_sets_reg)
  #   # df_test_preds_selected_pivot <- list()
  #   df_test_preds_list <- list()
  #   # train_mse_list <- list()
  #   
  #   parameter_values <- seq(1, 15, 2); l <- length(parameter_values)
  #   bias_sq <- numeric(l); variance <- numeric(l); training_mse <- numeric(l)
  #   
  #   for(count in seq_along(parameter_values))
  #   {
  #     d <- parameter_values[count]
  #     
  #     # FOR AVERAGE TRAINING AND TEST MSE FROM REPLICATED DATASETS AND TEST SET
  #     for(j in 1:num_rep_sets_reg)
  #     {
  #       
  #       # fit model of d degrees of freedom to each replicated (training) set
  #       # get predictions (y_hat's) for each replicated (training) set and test set
  #       y_rep <- df_training_ys[,j]; x_rep <- df_training_xs
  #       y_hat_training <- Rfast::knn(xnew = matrix(x_rep), y = y_rep, 
  #                                    x = matrix(x_rep), k = d, type = "R")
  #       y_hat_test <- Rfast::knn(xnew = matrix(df_test$x), y = y_rep, 
  #                                x = matrix(x_rep), k = d, type = "R")
  #       
  #       # store predictions for each set
  #       df_training_preds[,j] <- y_hat_training
  #       df_test_preds[,j] <- y_hat_test
  #       
  #       # calculate training MSE for each replicated (training) dataset
  #       train_mse[j] <- mean((y_rep-y_hat_training)^2)
  #       
  #     }
  #     
  #     df_test_preds_list[[count]] <- df_test_preds
  #     
  #     # calculate bias squared and variance for test MSE
  #     E_y_hat <- apply(df_test_preds, 1, mean)
  #     V_y_hat <- apply(df_test_preds, 1, var)
  #     bias_squared <- (E_y_hat - df_test$fx)^2
  #     
  #     # store bias squared, variance, and training MSE values
  #     training_mse[count] <- mean(train_mse)
  #     bias_sq[count] <- mean(bias_squared)
  #     variance[count] <- mean(V_y_hat)
  #     
  #     
  #   }
  #   
  #   
  #   # store results
  #   results <- data.frame(parameter_values = factor(parameter_values),
  #                         Bias_sq = bias_sq,
  #                         Variance = variance,
  #                         Test = bias_sq + variance + (noise_sd^2),
  #                         Training = training_mse)
  #   
  #   # return(list(results, parameter_values, df_test_preds_selected_pivot, noise_sd))
  #   return(list(results, parameter_values, df_test_preds_list, noise_sd))
  #   
  # })
  # 
  # df_bvt_calc_tree_reg <- reactive({
  #   
  #   df_test <- df_data_regression()[[2]]
  #   df_training_xs <- df_data_regression()[[3]]
  #   df_training_ys <- df_data_regression()[[4]]
  #   noise_sd <- df_data_regression()[[5]]
  #   
  #   num_training_points <- input$num_points_polynomial
  #   # ; noise_sd <- input$epsilon_polynomial
  #   
  #   df_training_preds <- matrix(nrow = num_training_points, ncol = num_rep_sets_reg)
  #   df_test_preds <- matrix(nrow = num_test_points_reg, ncol = num_rep_sets_reg)
  #   train_mse <- rep(NA, num_rep_sets_reg)
  #   # df_test_preds_selected_pivot <- list()
  #   df_test_preds_list <- list()
  #   # train_mse_list <- list()
  #   
  #   parameter_values <- seq(2, 16, 2); l <- length(parameter_values)
  #   bias_sq <- numeric(l); variance <- numeric(l); training_mse <- numeric(l)
  #   
  #   for(count in seq_along(parameter_values))
  #   {
  #     d <- parameter_values[count]
  #     
  #     # FOR AVERAGE TRAINING AND TEST MSE FROM REPLICATED DATASETS AND TEST SET
  #     for(j in 1:num_rep_sets_reg)
  #     {
  #       
  #       # fit model of d degrees of freedom to each replicated (training) set
  #       # get predictions (y_hat's) for each replicated (training) set and test set
  #       y_rep <- df_training_ys[,j]; x_rep <- df_training_xs
  #       m <- rpart(y_rep ~ x_rep, method = "anova", 
  #                  control = rpart.control(cp = 0, xval = 0, minbucket = 1, maxdepth = d))
  #       y_hat_training <- predict(m, newdata = data.frame(x_rep))
  #       y_hat_test <- predict(m, newdata = data.frame(x_rep = df_test$x))
  #       
  #       # store predictions for each set
  #       df_training_preds[,j] <- y_hat_training
  #       df_test_preds[,j] <- y_hat_test
  #       
  #       # calculate training MSE for each replicated (training) dataset
  #       train_mse[j] <- mean((y_rep-y_hat_training)^2)
  #       
  #     }
  #     
  #     df_test_preds_list[[count]] <- df_test_preds
  #     
  #     # calculate bias squared and variance for test MSE
  #     E_y_hat <- apply(df_test_preds, 1, mean)
  #     V_y_hat <- apply(df_test_preds, 1, var)
  #     bias_squared <- (E_y_hat - df_test$fx)^2
  #     
  #     # store bias squared, variance, and training MSE values
  #     training_mse[count] <- mean(train_mse)
  #     bias_sq[count] <- mean(bias_squared)
  #     variance[count] <- mean(V_y_hat)
  #     
  #     
  #   }
  #   
  #   
  #   # store results
  #   results <- data.frame(parameter_values = factor(parameter_values),
  #                         Bias_sq = bias_sq,
  #                         Variance = variance,
  #                         Test = bias_sq + variance + (noise_sd^2),
  #                         Training = training_mse)
  #   
  #   
  #   # return(list(results, parameter_values, df_test_preds_selected_pivot, noise_sd))
  #   return(list(results, parameter_values, df_test_preds_list, noise_sd))
  #   
  # })
  # 
  # df_bvt_calc_lasso <- reactive({
  #   
  #   df_test <- df_data_regression()[[2]]
  #   df_training_xs <- df_data_regression()[[3]]
  #   df_training_ys <- df_data_regression()[[4]]
  #   noise_sd <- df_data_regression()[[5]]
  #   
  #   num_training_points <- input$num_points_polynomial
  #   # ; noise_sd <- input$epsilon_polynomial
  #   
  #   poly_degree <- 5
  #   df_training_preds <- matrix(nrow = num_training_points, ncol = num_rep_sets_reg)
  #   df_test_preds <- matrix(nrow = num_test_points_reg, ncol = num_rep_sets_reg)
  #   train_mse <- rep(NA, num_rep_sets_reg)
  #   # df_test_preds_selected_pivot <- list()
  #   df_test_preds_list <- list()
  #   # train_mse_list <- list()
  #   model_coefficients <- list()
  #   
  #   # parameter_values <- c(0, 10^seq(-3,1,1))
  #   parameter_values <- 10^seq(-4,3,1)
  #   l <- length(parameter_values)
  #   bias_sq <- numeric(l); variance <- numeric(l); training_mse <- numeric(l)
  #   
  #   for(count in seq_along(parameter_values))
  #   {
  #     d <- parameter_values[count]
  #     m_coefs <- matrix(nrow = num_rep_sets_reg, ncol = poly_degree)
  #     
  #     # FOR AVERAGE TRAINING AND TEST MSE FROM REPLICATED DATASETS AND TEST SET
  #     for(j in 1:num_rep_sets_reg)
  #     {
  #       
  #       # fit model of d degrees of freedom to each replicated (training) set
  #       # get predictions (y_hat's) for each replicated (training) set and test set
  #       y_rep <- df_training_ys[,j]; x_rep <- df_training_xs
  #       
  #       x_rep_mat <- model.matrix(~ poly(x_rep, degree = poly_degree, raw = TRUE))[,-1]
  #       m <- glmnet(x = x_rep_mat, y = y_rep, family = "gaussian", alpha = 0, lambda = d)
  #       y_hat_training <- predict(m, newx = x_rep_mat)
  #       y_hat_test <- predict(m, newx = model.matrix(~ poly(df_test$x, degree = poly_degree, raw = TRUE))[,-1])
  #       m_coefs[j,] <- as.vector(m$beta)
  #       # }
  #       
  #       # store predictions for each set
  #       df_training_preds[,j] <- y_hat_training
  #       df_test_preds[,j] <- y_hat_test
  #       
  #       # calculate training MSE for each replicated (training) dataset
  #       train_mse[j] <- mean((y_rep-y_hat_training)^2)
  #       
  #     }
  #     
  #     df_test_preds_list[[count]] <- df_test_preds
  #     model_coefficients[[count]] <- m_coefs
  # 
  #     # calculate bias squared and variance for test MSE
  #     E_y_hat <- apply(df_test_preds, 1, mean)
  #     V_y_hat <- apply(df_test_preds, 1, var)
  #     bias_squared <- (E_y_hat - df_test$fx)^2
  #     
  #     # store bias squared, variance, and training MSE values
  #     training_mse[count] <- mean(train_mse)
  #     bias_sq[count] <- mean(bias_squared)
  #     variance[count] <- mean(V_y_hat)
  #     
  #     
  #   }
  #   
  #   
  #   # store results
  #   results <- data.frame(parameter_values = factor(log10(parameter_values)),
  #                         Bias_sq = bias_sq,
  #                         Variance = variance,
  #                         Test = bias_sq + variance + (noise_sd^2),
  #                         Training = training_mse)
  #   
  # 
  #   # return(list(results, parameter_values, df_test_preds_selected_pivot, noise_sd))
  #   return(list(results, parameter_values, df_test_preds_list, noise_sd, model_coefficients))
  #   
  # })
  
  
  
  
  
  

  

  
  ###########################
  # plots in first tab - Model Fit
  
  cb_pallete <- colorblind_pal()(8)
  
  output$plot_polynomial_p1 <- renderPlotly({
    
    # df_training <- df_data_regression()$df_training_full
    df_training <- df_data_regression()[[1]]
    
    # parameter_selected <- as.numeric(input$degrees)
    
    training_data_plot <- ggplot(data = df_training, aes(x=x, y=y_orig)) +
      geom_point(alpha = 0.4, size = 1) +
      stat_smooth(method = "lm", se = FALSE,
                  formula = y ~ poly(x, as.numeric(input$degrees_1), raw = TRUE), color = cb_pallete[7], linewidth = 0.75) +
      stat_smooth(method = "lm", se = FALSE,
                  formula = y ~ poly(x, as.numeric(input$degrees_2), raw = TRUE), color = cb_pallete[5], linewidth = 0.75) +
      stat_smooth(method = "lm", se = FALSE,
                  formula = y ~ poly(x, as.numeric(input$degrees_3), raw = TRUE), color = cb_pallete[3], linewidth = 0.75) +
      theme_bw() +
      labs(x = "Predictor x", y = "Response y", title = "Training Data")
    
    if(input$plot_true_polynomial == TRUE){ggplotly(training_data_plot + geom_line(aes(x = x, y = fx), color = cb_pallete[1], linewidth = 1))}
    else{ggplotly(training_data_plot)}}) %>%
    bindCache(input$degrees_1, input$degrees_2, input$degrees_3, input$plot_true_polynomial, df_data_regression())
  # bindCache(input$dataset_polynomial, input$num_points_polynomial, input$epsilon_polynomial, input$degrees_1, input$degrees_2, input$degrees_3, input$plot_true_polynomial, df_data_regression())
  # , cacheKeyExpr = { list(input$dataset_polynomial, input$num_points_polynomial, input$epsilon_polynomial, input$degrees_1, input$degrees_2, input$degrees_3, input$plot_true_polynomial, df_data_regression()) })
  
  output$plot_spline_p1 <- renderPlotly({
    
    
    # df_training <- df_data_regression()$df_training_full
    df_training <- df_data_regression()[[1]]
    
    # parameter_selected <- as.numeric(input$degrees_of_freedom)
    
    training_data_plot <- ggplot(data = df_training, aes(x=x, y=y_orig)) + 
      geom_point(alpha = 0.4, size = 1) + 
      geom_spline(df = as.numeric(input$degrees_of_freedom_1), color = cb_pallete[7], linewidth = 0.75) +
      geom_spline(df = as.numeric(input$degrees_of_freedom_2), color = cb_pallete[5], linewidth = 0.75) +
      geom_spline(df = as.numeric(input$degrees_of_freedom_3), color = cb_pallete[3], linewidth = 0.75) +
      theme_bw() +
      labs(x = "Predictor x", y = "Response y", title = "Training Data")
    
    if(input$plot_true_spline == TRUE){ggplotly(training_data_plot + geom_line(aes(x = x, y = fx), color = cb_pallete[1], linewidth = 1))}
    else{ggplotly(training_data_plot)}}) %>% 
    bindCache(input$degrees_of_freedom_1, input$degrees_of_freedom_2, input$degrees_of_freedom_3, input$plot_true_spline, df_data_regression())
  # , cacheKeyExpr = { list(input$dataset_polynomial, input$num_points_polynomial, input$epsilon_polynomial, input$degrees_of_freedom_1, input$degrees_of_freedom_2, input$degrees_of_freedom_3, input$plot_true_spline, df_data_regression()) })
  
  output$plot_knn_reg_p1 <- renderPlotly({
    
    # df_training <- df_data_regression()$df_training_full
    df_training <- df_data_regression()[[1]]
    
    # parameter_selected <- as.numeric(input$k_values)
    
    x_seq <- seq(min(df_training$x, na.rm = TRUE), max(df_training$x, na.rm = TRUE), 0.001)
    knnfit_1 <- knnreg(y_orig ~ x, data = df_training, k = as.numeric(input$k_values_1))
    knnfit_2 <- knnreg(y_orig ~ x, data = df_training, k = as.numeric(input$k_values_2))
    knnfit_3 <- knnreg(y_orig ~ x, data = df_training, k = as.numeric(input$k_values_3))
    # preds <- predict(knnfit, newdata = data.frame(x = x_seq))
    predictions <- data.frame(x_seq, preds_1 = predict(knnfit_1, newdata = data.frame(x = x_seq)),
                              preds_2 = predict(knnfit_2, newdata = data.frame(x = x_seq)),
                              preds_3 = predict(knnfit_3, newdata = data.frame(x = x_seq)))
    
    training_data_plot <- ggplot(data = df_training, aes(x = x, y = y_orig)) +
      geom_point(alpha = 0.4, size = 1) +
      geom_line(data = predictions, aes(x = x_seq, y = preds_1), color = cb_pallete[7], linewidth = 0.5) +
      geom_line(data = predictions, aes(x = x_seq, y = preds_2), color = cb_pallete[5], linewidth = 0.5) +
      geom_line(data = predictions, aes(x = x_seq, y = preds_3), color = cb_pallete[3], linewidth = 0.5) +
      theme_bw() +
      labs(x = "Predictor x", y = "Response y", title = "Training Data")
    
    if(input$plot_true_knn_reg == TRUE){ggplotly(training_data_plot + geom_line(aes(x = x, y = fx), color = cb_pallete[1], linewidth = 1))}
    else{ggplotly(training_data_plot)}}) %>% 
    bindCache(input$k_values_1, input$k_values_2, input$k_values_3, input$plot_true_knn_reg, df_data_regression())
  # , cacheKeyExpr = { list(input$dataset_polynomial, input$num_points_polynomial, input$epsilon_polynomial, input$k_values_1, input$k_values_2, input$k_values_3, input$plot_true_knn_reg, df_data_regression()) }) 
  
  output$plot_tree_reg_p1a <- renderPlotly({
    
    # df_training <- df_data_regression()$df_training_full
    df_training <- df_data_regression()[[1]]
    
    # parameter_selected <- as.numeric(input$depths)
    
    x_seq <- seq(min(df_training$x, na.rm = TRUE), max(df_training$x, na.rm = TRUE), 0.001)
    treefit_1 <- rpart(y_orig ~ x, data = df_training, method = "anova",
                       control = rpart.control(cp = 0, xval = 0,
                                               minbucket = 1, maxdepth = input$depths_1))
    treefit_2 <- rpart(y_orig ~ x, data = df_training, method = "anova",
                       control = rpart.control(cp = 0, xval = 0,
                                               minbucket = 1, maxdepth = input$depths_2))
    treefit_3 <- rpart(y_orig ~ x, data = df_training, method = "anova",
                       control = rpart.control(cp = 0, xval = 0,
                                               minbucket = 1, maxdepth = input$depths_3))
    # preds <- predict(treefit, newdata = data.frame(x = x_seq))
    predictions <- data.frame(x_seq, preds_1 = predict(treefit_1, newdata = data.frame(x = x_seq)),
                              preds_2 = predict(treefit_2, newdata = data.frame(x = x_seq)),
                              preds_3 = predict(treefit_3, newdata = data.frame(x = x_seq)))
    
    training_data_plot <- ggplot(data = df_training, aes(x = x, y = y_orig)) +
      geom_point(alpha = 0.4, size = 1) +
      geom_line(data = predictions, aes(x = x_seq, y = preds_1), color = cb_pallete[7], linewidth = 0.5) +
      geom_line(data = predictions, aes(x = x_seq, y = preds_2), color = cb_pallete[5], linewidth = 0.5) +
      geom_line(data = predictions, aes(x = x_seq, y = preds_3), color = cb_pallete[3], linewidth = 0.5) +
      theme_bw() +
      labs(x = "Predictor x", y = "Response y", title = "Training Data")
    
    if(input$plot_true_tree_reg == TRUE){ggplotly(training_data_plot + geom_line(aes(x = x, y = fx), color = cb_pallete[1], linewidth = 1))}
    else{ggplotly(training_data_plot)}
  }) %>% 
    bindCache(input$depths_1, input$depths_2, input$depths_3, input$plot_true_tree_reg, df_data_regression())
  # , cacheKeyExpr = { list(input$dataset_polynomial, input$num_points_polynomial, input$epsilon_polynomial,  input$depths_1, input$depths_2, input$depths_3, input$plot_true_tree_reg, df_data_regression()) }) 
  # 
  # output$plot_tree_reg_p1b <- renderCachedPlot({
  #   
  #   df_training <- df_data_regression()[[1]]
  #   
  #   parameter_selected <- as.numeric(input$depths)
  #   
  #   x_seq <- seq(min(df_training$x, na.rm = TRUE), max(df_training$x, na.rm = TRUE), 0.001)
  #   treefit <- rpart(y_orig ~ x, data = df_training, method = "anova",
  #                    control = rpart.control(cp = 0, xval = 0,
  #                                            minbucket = 1, maxdepth = parameter_selected))
  #   preds <- predict(treefit, newdata = data.frame(x = x_seq))
  #   predictions <- data.frame(x_seq, preds)
  #   
  #   rpart.plot(treefit)
  # }, cacheKeyExpr = { list(input$dataset_polynomial, input$num_points_polynomial, input$epsilon_polynomial, input$plot_true_knn_reg, df_data_regression()) }) 
  # 
  output$plot_lasso_p1 <- renderPlotly({
    
    # df_training <- df_data_regression()$df_training_full
    df_training <- df_data_regression()[[1]]
    poly_degree <- 5
    
    # parameter_selected <- as.numeric(input$lambda_lasso)
    
    regularized_fit_1 <- glmnet(x = model.matrix(~ poly(df_training$x, degree = poly_degree, raw = TRUE))[,-1], 
                                y = df_training$y_orig,
                                alpha = 0,
                                lambda = 10^as.numeric(input$lambda_lasso_1))
    df_training$preds_1 <- predict(regularized_fit_1, newx = model.matrix(~ poly(df_training$x, degree = poly_degree, raw = TRUE))[,-1])
    # }
    
    regularized_fit_2 <- glmnet(x = model.matrix(~ poly(df_training$x, degree = poly_degree, raw = TRUE))[,-1], 
                                y = df_training$y_orig,
                                alpha = 0,
                                lambda = 10^as.numeric(input$lambda_lasso_2))
    df_training$preds_2 <- predict(regularized_fit_2, newx = model.matrix(~ poly(df_training$x, degree = poly_degree, raw = TRUE))[,-1])
    
    
    regularized_fit_3 <- glmnet(x = model.matrix(~ poly(df_training$x, degree = poly_degree, raw = TRUE))[,-1], 
                                y = df_training$y_orig,
                                alpha = 0,
                                lambda = 10^as.numeric(input$lambda_lasso_3))
    df_training$preds_3 <- predict(regularized_fit_3, newx = model.matrix(~ poly(df_training$x, degree = poly_degree, raw = TRUE))[,-1])
    
    
    training_data_plot <- ggplot(data = df_training, aes(x = x, y = y_orig)) +
      geom_point(alpha = 0.4, size = 1) +
      geom_line(aes(x = x, y = preds_1), color = cb_pallete[7], linewidth = 0.5) +
      geom_line(aes(x = x, y = preds_2), color = cb_pallete[5], linewidth = 0.5) +
      geom_line(aes(x = x, y = preds_3), color = cb_pallete[3], linewidth = 0.5) +
      theme_bw() +
      labs(x = "Predictor x", y = "Response y", title = "Training Data")
    
    if(input$plot_true_lasso == TRUE){ggplotly(training_data_plot + geom_line(aes(x = x, y = fx), color = cb_pallete[1], linewidth = 1))}
    else{ggplotly(training_data_plot)}
  }) %>% 
    bindCache(input$lambda_lasso_1, input$lambda_lasso_2, input$lambda_lasso_3, input$plot_true_lasso, df_data_regression())
  # , cacheKeyExpr = { list(input$dataset_polynomial, input$num_points_polynomial, input$epsilon_polynomial, input$lambda_lasso_1, input$lambda_lasso_2, input$lambda_lasso_3, input$plot_true_lasso, df_data_regression()) }) 
  
  output$plot_rf_p1 <- renderPlotly({
    
    # df_training <- df_data_regression()$df_training_full
    df_training <- df_data_regression()[[1]]
    
    x_seq <- seq(min(df_training$x, na.rm = TRUE), max(df_training$x, na.rm = TRUE), 0.001)
    
    rffit <- ranger(formula = y_orig ~ x, data = df_training,
                    # x = data.frame(df_training_subset$x), y = df_training_subset$y_orig, xtest = data.frame(x_seq), #data = data.frame(y_rep = y_rep, x_rep = x_rep),
                    num.trees = 10, # number of trees to grow (bootstrap samples) usually 500
                    mtry = 1, 
                    min.bucket = 1,
                    keep.inbag=TRUE, seed = 208)
    
    predictions <- data.frame(x_seq, preds_1 = predict(rffit, data = data.frame(x = x_seq))$predictions)
    
    training_data_plot <- ggplot(data = df_training, aes(x = x, y = y_orig)) +
      geom_point(alpha = 0.4, size = 1) +
      geom_line(data = predictions, aes(x = x_seq, y = preds_1), color = cb_pallete[7], linewidth = 0.5) +
      theme_bw() +
      labs(x = "Predictor x", y = "Response y", title = "Training Data")
    
    if(input$plot_true_rf_1 == TRUE){ggplotly(training_data_plot + geom_line(aes(x = x, y = fx), color = cb_pallete[1], linewidth = 1))}
    else{ggplotly(training_data_plot)}
  }) %>% 
    bindCache(input$plot_true_rf_1, df_data_regression())
  # , cacheKeyExpr = { list(input$dataset_polynomial, input$num_points_polynomial, input$epsilon_polynomial, input$plot_true_rf_1, df_data_regression()) }) 
  
  
  ###########################
  # plots in second tab - Bias-Variance Trade-Off and Bootsrapping
  
  output$plot_polynomial_p2 <- renderPlotly({
    
    # df_data_regression()$allplots2_polynomial
    # plots <- second_row_plots_m(df_data = df_bvt_calc_polynomial(), x_axis_lab = "Degree")
    # second_row_plots_m(df_data = df_bvt_calc_polynomial(), x_axis_lab = "Degree")
    second_row_plots(df_data = df_data_regression()[[3]], x_axis_lab = "Degree")
    
    # (plots[[1]] | plots[[2]] | plots[[3]] | plots[[4]])
    # subplot(plots[[1]], plots[[2]], plots[[3]], plots[[4]], nrows = 1, shareX = TRUE)
  })  %>% bindCache(df_data_regression())
    # bindCache(df_bvt_calc_polynomial())
  # , cacheKeyExpr = { list(input$dataset_polynomial, input$num_points_polynomial, input$epsilon_polynomial, df_data_regression(), df_bvt_calc_polynomial()) }) 
  
  output$plot_spline_p2 <- renderPlotly({
    
    # df_data_regression()$allplots2_spline
    # second_row_plots_m(df_data = df_bvt_calc_spline(), x_axis_lab = "Degrees of Freedom")
    second_row_plots(df_data = df_data_regression()[[4]], x_axis_lab = "Degrees of Freedom")
    
    
    # (plots[[1]] | plots[[2]] | plots[[3]] | plots[[4]]) 
    # subplot(plots[[1]], plots[[2]], plots[[3]], plots[[4]], nrows = 1, shareX = TRUE)
  }) %>% bindCache(df_data_regression())
    # bindCache(df_bvt_calc_spline())
  # , cacheKeyExpr = { list(input$dataset_polynomial, input$num_points_polynomial, input$epsilon_polynomial, df_data_regression(), df_bvt_calc_spline()) }) 
  
  output$plot_knn_reg_p2 <- renderPlotly({
    
    # df_data_regression()$allplots2_knn_reg
    # second_row_plots_m(df_data = df_bvt_calc_knn_reg(), x_axis_lab = "K")
    second_row_plots(df_data = df_data_regression()[[5]], x_axis_lab = "K")
    
    
    # (plots[[1]] | plots[[2]] | plots[[3]] | plots[[4]]) 
    # subplot(plots[[1]], plots[[2]], plots[[3]], plots[[4]], nrows = 1, shareX = TRUE)
    
  }) %>% bindCache(df_data_regression())
    # bindCache(df_bvt_calc_knn_reg())
  # , cacheKeyExpr = { list(input$dataset_polynomial, input$num_points_polynomial, input$epsilon_polynomial, df_data_regression(), df_bvt_calc_knn_reg()) }) 

  output$plot_tree_reg_p2 <- renderPlotly({
    
    # df_data_regression()$allplots2_tree_reg
    # second_row_plots_m(df_data = df_bvt_calc_tree_reg(), x_axis_lab = "Tree Depth")
    second_row_plots(df_data = df_data_regression()[[6]], x_axis_lab = "Tree Depth")
    
    
    # (plots[[1]] | plots[[2]] | plots[[3]] | plots[[4]]) 
    # subplot(plots[[1]], plots[[2]], plots[[3]], plots[[4]], nrows = 1, shareX = TRUE)
    
  }) %>% bindCache(df_data_regression())
    # bindCache(df_bvt_calc_tree_reg())
  # , cacheKeyExpr = { list(input$dataset_polynomial, input$num_points_polynomial, input$epsilon_polynomial, df_data_regression(), df_bvt_calc_tree_reg()) }) 

  output$plot_lasso_p2 <- renderPlotly({
    
    # df_data_regression()$allplots2_lasso
    # second_row_plots_m(df_data = df_bvt_calc_lasso(), x_axis_lab = "log10(\u03bb)")
    second_row_plots(df_data = df_data_regression()[[7]], x_axis_lab = "log10(\u03bb)")
    
    
    # (plots[[1]] | plots[[2]] | plots[[3]] | plots[[4]]) 
    # subplot(plots[[1]], plots[[2]], plots[[3]], plots[[4]], nrows = 1, shareX = TRUE)
    
  }) %>% bindCache(df_data_regression())
    # bindCache(df_bvt_calc_lasso())
  # , cacheKeyExpr = { list(input$dataset_polynomial, input$num_points_polynomial, input$epsilon_polynomial, df_data_regression(), df_bvt_calc_lasso()) }) 
  
  output$plot_rf_p2 <- renderCachedPlot({
    
    df_training_subset <- df_bvt_calc_rf()[[1]]
    x_seq <- df_bvt_calc_rf()[[2]]
    rffit <- df_bvt_calc_rf()[[3]]
    preds_rf <- df_bvt_calc_rf()[[4]]
    
    # trying to replicate F.7.1b from Lindholm et al. (could use to demonstrating bootstrapping, could use without true funtion and individual tree fits)
    ylimit <- c(min(df_training_subset$y_orig) - 0.3, max(df_training_subset$y_orig) + 0.5)
    xlimit <- c(min(df_training_subset$x) - 0.3, max(df_training_subset$x) + 0.3)
    
    g1a <- ggplot(data = inner_join(data.frame(Var1=rownames(df_training_subset), Freq=rffit$inbag.counts[[1]]) %>% filter(Freq!=0) %>% mutate(Freq = as.factor(Freq)), df_training_subset), aes(x = x, y = y_orig)) +
      geom_point(
        alpha = 0.4,
        # size = 2,
        aes(size = Freq)) + 
      geom_text(aes(label = ifelse(Freq != 1, Freq, NA)), size = 3) +
      theme_bw() + xlim(xlimit) + ylim(ylimit) + theme(legend.position = "none") +
      labs(x = "Predictor x", y = "Response y", title = "Bootstrapped dataset 1")
    g2a <- ggplot(data = inner_join(data.frame(Var1=rownames(df_training_subset), Freq=rffit$inbag.counts[[2]]) %>% filter(Freq!=0) %>% mutate(Freq = as.factor(Freq)), df_training_subset), aes(x = x, y = y_orig)) +
      geom_point(
        alpha = 0.4,
        # size = 2,
        aes(size = Freq)) + 
      geom_text(aes(label = ifelse(Freq != 1, Freq, NA)), size = 3) +
      theme_bw() + xlim(xlimit) + ylim(ylimit) + theme(legend.position = "none") +
      labs(x = "Predictor x", y = "Response y", title = "Bootstrapped dataset 2")
    g3a <- ggplot(data = inner_join(data.frame(Var1=rownames(df_training_subset), Freq=rffit$inbag.counts[[3]]) %>% filter(Freq!=0) %>% mutate(Freq = as.factor(Freq)), df_training_subset), aes(x = x, y = y_orig)) +
      geom_point(
        alpha = 0.4,
        # size = 2,
        aes(size = Freq)) + 
      geom_text(aes(label = ifelse(Freq != 1, Freq, NA)), size = 3) +
      theme_bw() + xlim(xlimit) + ylim(ylimit) + theme(legend.position = "none") +
      labs(x = "Predictor x", y = "Response y", title = "Bootstrapped dataset 3")
    g4a <- ggplot(data = inner_join(data.frame(Var1=rownames(df_training_subset), Freq=rffit$inbag.counts[[4]]) %>% filter(Freq!=0) %>% mutate(Freq = as.factor(Freq)), df_training_subset), aes(x = x, y = y_orig)) +
      geom_point(
        alpha = 0.4,
        # size = 2,
        aes(size = Freq)) + 
      geom_text(aes(label = ifelse(Freq != 1, Freq, NA)), size = 3) +
      theme_bw() + xlim(xlimit) + ylim(ylimit) + theme(legend.position = "none") +
      labs(x = "Predictor x", y = "Response y", title = "Bootstrapped dataset 4")
    # g5a <- ggplot(data = inner_join(data.frame(Var1=rownames(df_training_subset), Freq=rffit$inbag.counts[[5]]) %>% filter(Freq!=0) %>% mutate(Freq = as.factor(Freq)), df_training_subset), aes(x = x, y = y_orig)) +
    #   geom_point(
    #     alpha = 0.4,
    #     # size = 2,
    #     aes(size = Freq)) + 
    #   geom_text(aes(label = ifelse(Freq != 1, Freq, NA)), size = 3) +
    #   theme_bw() + xlim(xlimit) + ylim(ylimit) + theme(legend.position = "none") +
    #   labs(x = "Predictor x", y = "Response y", title = "Bootstrapped dataset 5")
    # g6a <- ggplot(data = inner_join(data.frame(Var1=rownames(df_training_subset), Freq=rffit$inbag.counts[[6]]) %>% filter(Freq!=0) %>% mutate(Freq = as.factor(Freq)), df_training_subset), aes(x = x, y = y_orig)) +
    #   geom_point(
    #     alpha = 0.4,
    #     # size = 2,
    #     aes(size = Freq)) + 
    #   geom_text(aes(label = ifelse(Freq != 1, Freq, NA)), size = 3) +
    #   theme_bw() + xlim(xlimit) + ylim(ylimit) + theme(legend.position = "none") +
    #   labs(x = "Predictor x", y = "Response y", title = "Bootstrapped dataset 6")
    
    g1b <- g1a + geom_line(data = data.frame(x_seq, y = preds_rf$predictions[,1]), aes(x = x_seq, y = y), color = cb_pallete[7], linewidth = 1)
    g2b <- g2a + geom_line(data = data.frame(x_seq, y = preds_rf$predictions[,2]), aes(x = x_seq, y = y), color = cb_pallete[7], linewidth = 1)
    g3b <- g3a + geom_line(data = data.frame(x_seq, y = preds_rf$predictions[,3]), aes(x = x_seq, y = y), color = cb_pallete[7], linewidth = 1)
    g4b <- g4a + geom_line(data = data.frame(x_seq, y = preds_rf$predictions[,4]), aes(x = x_seq, y = y), color = cb_pallete[7], linewidth = 1)
    # g5b <- g5a + geom_line(data = data.frame(x_seq, y = preds_rf$predictions[,5]), aes(x = x_seq, y = y), color = cb_pallete[7], linewidth = 1)
    # g6b <- g6a + geom_line(data = data.frame(x_seq, y = preds_rf$predictions[,6]), aes(x = x_seq, y = y), color = cb_pallete[7], linewidth = 1)
    
    # (g1b | g2b | g3b)/(g4b | g5b | g6b)
    
    g1c <- g1a + geom_line(data = df_training_subset, aes(x = x, y = fx), color = cb_pallete[1], linewidth = 1)
    g2c <- g2a + geom_line(data = df_training_subset, aes(x = x, y = fx), color = cb_pallete[1], linewidth = 1)
    g3c <- g3a + geom_line(data = df_training_subset, aes(x = x, y = fx), color = cb_pallete[1], linewidth = 1)
    g4c <- g4a + geom_line(data = df_training_subset, aes(x = x, y = fx), color = cb_pallete[1], linewidth = 1)
    # g5c <- g5a + geom_line(data = df_training_subset, aes(x = x, y = fx), color = cb_pallete[1], linewidth = 1)
    # g6c <- g6a + geom_line(data = df_training_subset, aes(x = x, y = fx), color = cb_pallete[1], linewidth = 1)
    
    # (g1c | g2c | g3c)/(g4c | g5c | g6c)
    
    g1d <- g1a + geom_line(data = data.frame(x_seq, y = preds_rf$predictions[,1]), aes(x = x_seq, y = y), color = cb_pallete[7], linewidth = 1) + geom_line(data = df_training_subset, aes(x = x, y = fx), color = cb_pallete[1], linewidth = 1)
    g2d <- g2a + geom_line(data = data.frame(x_seq, y = preds_rf$predictions[,2]), aes(x = x_seq, y = y), color = cb_pallete[7], linewidth = 1) + geom_line(data = df_training_subset, aes(x = x, y = fx), color = cb_pallete[1], linewidth = 1)
    g3d <- g3a + geom_line(data = data.frame(x_seq, y = preds_rf$predictions[,3]), aes(x = x_seq, y = y), color = cb_pallete[7], linewidth = 1) + geom_line(data = df_training_subset, aes(x = x, y = fx), color = cb_pallete[1], linewidth = 1)
    g4d <- g4a + geom_line(data = data.frame(x_seq, y = preds_rf$predictions[,4]), aes(x = x_seq, y = y), color = cb_pallete[7], linewidth = 1) + geom_line(data = df_training_subset, aes(x = x, y = fx), color = cb_pallete[1], linewidth = 1)
    # g5d <- g5a + geom_line(data = data.frame(x_seq, y = preds_rf$predictions[,5]), aes(x = x_seq, y = y), color = cb_pallete[7], linewidth = 1) + geom_line(data = df_training_subset, aes(x = x, y = fx), color = cb_pallete[1], linewidth = 1)
    # g6d <- g6a + geom_line(data = data.frame(x_seq, y = preds_rf$predictions[,6]), aes(x = x_seq, y = y), color = cb_pallete[7], linewidth = 1) + geom_line(data = df_training_subset, aes(x = x, y = fx), color = cb_pallete[1], linewidth = 1)
    
    
    
    # if(input$plot_rf_trees == TRUE & input$plot_true_rf == FALSE){(g1b | g2b | g3b)/(g4b | g5b | g6b)}
    # else if(input$plot_rf_trees == FALSE & input$plot_true_rf == TRUE){(g1c | g2c | g3c)/(g4c | g5c | g6c)}
    # else if(input$plot_rf_trees == TRUE & input$plot_true_rf == TRUE){(g1d | g2d | g3d)/(g4d | g5d | g6d)}
    # else if(input$plot_rf_trees == FALSE & input$plot_true_rf == FALSE){(g1a | g2a | g3a)/(g4a | g5a | g6a)}
    
    if(input$plot_rf_trees == TRUE & input$plot_true_rf_2 == FALSE){(g1b | g2b)/(g3b | g4b)}
    else if(input$plot_rf_trees == FALSE & input$plot_true_rf_2 == TRUE){(g1c | g2c)/(g3c | g4c)}
    else if(input$plot_rf_trees == TRUE & input$plot_true_rf_2 == TRUE){(g1d | g2d)/(g3d | g4d)}
    else if(input$plot_rf_trees == FALSE & input$plot_true_rf_2 == FALSE){(g1a | g2a)/(g3a | g4a)}
    
  }, cacheKeyExpr = { list(df_bvt_calc_rf(), input$plot_rf_trees, input$plot_true_rf_2) })    
  
  
  ###########################
  # old code for plots, replaced by a function and code above
  
  # output$plot_polynomial_p2a <- renderPlot({
  #   
  #   results <- df_bvt_calc_polynomial()[[1]]
  #   noise_sd <- df_bvt_calc_polynomial()[[4]]
  # 
  #   results_mse <- results %>%
  #     pivot_longer(cols = 2:5, names_to = "mse_type", values_to = "mse")
  # 
  #   mse_plot <- ggplot(data = results_mse %>% filter(mse_type %in% c("Test", "Training"))) +
  #     geom_point(aes(x = parameter_values, y = mse, color = mse_type)) +
  #     geom_line(aes(x = parameter_values, y = mse, group = mse_type, color = mse_type)) +
  #     scale_color_colorblind(labels = c("Test MSE", "Training MSE")) +          # scale_color_manual(values = colorblind_pal()(6)[-c(1:3, 5)])
  #     geom_hline(yintercept = noise_sd^2, linetype = 2) + theme_bw() +
  #     theme(legend.position = "top", legend.title = element_blank()) +
  #     labs(y = "Mean Squared Error (MSE)", color = "MSE", x = "Degree") +
  #     ylim(c(min(results_mse$mse) - 0.03, max(results_mse$mse) + 0.03))
  # 
  #   mse_plot
  #   
  # })
  # 
  # output$plot_spline_p2a <- renderPlot({
  #   
  #   results <- df_bvt_calc_spline()[[1]]
  #   noise_sd <- df_bvt_calc_spline()[[4]]
  #   
  #   results_mse <- results %>%
  #     pivot_longer(cols = 2:5, names_to = "mse_type", values_to = "mse")
  #   
  #   mse_plot <- ggplot(data = results_mse %>% filter(mse_type %in% c("Test", "Training"))) +
  #     geom_point(aes(x = parameter_values, y = mse, color = mse_type)) +
  #     geom_line(aes(x = parameter_values, y = mse, group = mse_type, color = mse_type)) +
  #     scale_color_colorblind(labels = c("Test MSE", "Training MSE")) +          # scale_color_manual(values = colorblind_pal()(6)[-c(1:3, 5)])
  #     geom_hline(yintercept = noise_sd^2, linetype = 2) + theme_bw() +
  #     theme(legend.position = "top", legend.title = element_blank()) +
  #     labs(y = "Mean Squared Error (MSE)", color = "MSE", x = "Degrees of Freedom") +
  #     ylim(c(min(results_mse$mse) - 0.03, max(results_mse$mse) + 0.03))
  #   
  #   mse_plot
  #   
  #   
  # })
  # 
  # output$plot_knn_reg_p2a <- renderPlot({
  #   
  #   results <- df_bvt_calc_knn_reg()[[1]]
  #   noise_sd <- df_bvt_calc_knn_reg()[[4]]
  #   
  #   
  #   results_mse <- results %>%
  #     pivot_longer(cols = 2:5, names_to = "mse_type", values_to = "mse")
  #   
  #   mse_plot <- ggplot(data = results_mse %>% filter(mse_type %in% c("Test", "Training"))) +
  #     geom_point(aes(x = parameter_values, y = mse, color = mse_type)) +
  #     geom_line(aes(x = parameter_values, y = mse, group = mse_type, color = mse_type)) +
  #     scale_color_colorblind(labels = c("Test MSE", "Training MSE")) +          # scale_color_manual(values = colorblind_pal()(6)[-c(1:3, 5)])
  #     geom_hline(yintercept = noise_sd^2, linetype = 2) + theme_bw() +
  #     theme(legend.position = "top", legend.title = element_blank()) +
  #     labs(y = "Mean Squared Error (MSE)", color = "MSE", x = "K") +
  #     ylim(c(min(results_mse$mse) - 0.03, max(results_mse$mse) + 0.03))
  #   
  #   mse_plot
  #   
  # })
  # 
  # output$plot_tree_reg_p2a <- renderPlot({
  #   
  #   results <- df_bvt_calc_tree_reg()[[1]]
  #   noise_sd <- df_bvt_calc_tree_reg()[[4]]
  #   
  #   results_mse <- results %>%
  #     pivot_longer(cols = 2:5, names_to = "mse_type", values_to = "mse")
  #   
  #   mse_plot <- ggplot(data = results_mse %>% filter(mse_type %in% c("Test", "Training"))) +
  #     geom_point(aes(x = parameter_values, y = mse, color = mse_type)) +
  #     geom_line(aes(x = parameter_values, y = mse, group = mse_type, color = mse_type)) +
  #     scale_color_colorblind(labels = c("Test MSE", "Training MSE")) +          # scale_color_manual(values = colorblind_pal()(6)[-c(1:3, 5)])
  #     geom_hline(yintercept = noise_sd^2, linetype = 2) + theme_bw() +
  #     theme(legend.position = "top", legend.title = element_blank()) +
  #     labs(y = "Mean Squared Error (MSE)", color = "MSE", x = "Tree Depth") +
  #     ylim(c(min(results_mse$mse) - 0.03, max(results_mse$mse) + 0.03))
  #   
  #   mse_plot
  #   
  #   
  # })
  # 
  # output$plot_lasso_p2a <- renderPlot({
  #   
  #   results <- df_bvt_calc_lasso()[[1]]
  #   noise_sd <- df_bvt_calc_lasso()[[4]]
  #   
  #   results_mse <- results %>%
  #     pivot_longer(cols = 2:5, names_to = "mse_type", values_to = "mse")
  #   
  #   mse_plot <- ggplot(data = results_mse %>% filter(mse_type %in% c("Test", "Training"))) +
  #     geom_point(aes(x = parameter_values, y = mse, color = mse_type)) +
  #     geom_line(aes(x = parameter_values, y = mse, group = mse_type, color = mse_type)) +
  #     scale_color_colorblind(labels = c("Test MSE", "Training MSE")) +          # scale_color_manual(values = colorblind_pal()(6)[-c(1:3, 5)])
  #     geom_hline(yintercept = noise_sd^2, linetype = 2) + theme_bw() +
  #     theme(legend.position = "top", legend.title = element_blank()) +
  #     labs(y = "Mean Squared Error (MSE)", color = "MSE", x = bquote(log[10](lambda))) +
  #     ylim(c(min(results_mse$mse) - 0.03, max(results_mse$mse) + 0.03))
  #   
  #   mse_plot
  #   
  # 
  # })
  # 
  # 
  # 
  # output$plot_polynomial_p2b <- renderPlot({
  #   
  #   results <- df_bvt_calc_polynomial()[[1]]
  #   noise_sd <- df_bvt_calc_polynomial()[[4]]
  #   
  # 
  #   results_mse <- results %>%
  #     pivot_longer(cols = 2:5, names_to = "mse_type", values_to = "mse")
  #   
  #   bias_test_var_plot <- ggplot(data = results_mse %>% filter(mse_type != "Training")) +
  #     geom_point(aes(x = parameter_values, y = mse, color = mse_type)) +
  #     geom_line(aes(x = parameter_values, y = mse, group = mse_type, color = mse_type)) +
  #     scale_color_manual(values = cb_pallete[c(8, 1, 4)], labels = c(bquote(Bias^2), "Test MSE", "Variance")) +
  #     geom_hline(yintercept = noise_sd^2, linetype = 2) + theme_bw() +
  #     theme(legend.position = "top", legend.key.spacing.x = unit(0.1, 'cm'), legend.key.size = unit(0.5, "cm")) +
  #     labs(y = "", color = "", x = "Degree") +
  #     ylim(c(min(results_mse$mse) - 0.03, max(results_mse$mse) + 0.03))
  #   
  #   bias_test_var_plot
  #   
  # })
  # 
  # output$plot_spline_p2b <- renderPlot({
  #   
  #   results <- df_bvt_calc_spline()[[1]]
  #   noise_sd <- df_bvt_calc_spline()[[4]]
  #   
  #   
  #   results_mse <- results %>%
  #     pivot_longer(cols = 2:5, names_to = "mse_type", values_to = "mse")
  #   
  #   bias_test_var_plot <- ggplot(data = results_mse %>% filter(mse_type != "Training")) +
  #     geom_point(aes(x = parameter_values, y = mse, color = mse_type)) +
  #     geom_line(aes(x = parameter_values, y = mse, group = mse_type, color = mse_type)) +
  #     scale_color_manual(values = cb_pallete[c(8, 1, 4)], labels = c(bquote(Bias^2), "Test MSE", "Variance")) +
  #     geom_hline(yintercept = noise_sd^2, linetype = 2) + theme_bw() +
  #     theme(legend.position = "top", legend.key.spacing.x = unit(0.1, 'cm'), legend.key.size = unit(0.5, "cm")) +
  #     labs(y = "", color = "", x = "Degrees of Freedom") +
  #     ylim(c(min(results_mse$mse) - 0.03, max(results_mse$mse) + 0.03))
  #   
  #   bias_test_var_plot
  #   
  #   
  # })
  # 
  # output$plot_knn_reg_p2b <- renderPlot({
  #   
  #   results <- df_bvt_calc_knn_reg()[[1]]
  #   noise_sd <- df_bvt_calc_knn_reg()[[4]]
  #   
  #   results_mse <- results %>%
  #     pivot_longer(cols = 2:5, names_to = "mse_type", values_to = "mse")
  #   
  #   bias_test_var_plot <- ggplot(data = results_mse %>% filter(mse_type != "Training")) +
  #     geom_point(aes(x = parameter_values, y = mse, color = mse_type)) +
  #     geom_line(aes(x = parameter_values, y = mse, group = mse_type, color = mse_type)) +
  #     scale_color_manual(values = cb_pallete[c(8, 1, 4)], labels = c(bquote(Bias^2), "Test MSE", "Variance")) +
  #     geom_hline(yintercept = noise_sd^2, linetype = 2) + theme_bw() +
  #     theme(legend.position = "top", legend.key.spacing.x = unit(0.1, 'cm'), legend.key.size = unit(0.5, "cm")) +
  #     labs(y = "", color = "", x = "K") +
  #     ylim(c(min(results_mse$mse) - 0.03, max(results_mse$mse) + 0.03))
  #   
  #   bias_test_var_plot
  #   
  #   
  #   
  # })
  # 
  # output$plot_tree_reg_p2b <- renderPlot({
  #   
  #   results <- df_bvt_calc_tree_reg()[[1]]
  #   noise_sd <- df_bvt_calc_tree_reg()[[4]]
  #   
  #   results_mse <- results %>%
  #     pivot_longer(cols = 2:5, names_to = "mse_type", values_to = "mse")
  #   
  #   bias_test_var_plot <- ggplot(data = results_mse %>% filter(mse_type != "Training")) +
  #     geom_point(aes(x = parameter_values, y = mse, color = mse_type)) +
  #     geom_line(aes(x = parameter_values, y = mse, group = mse_type, color = mse_type)) +
  #     scale_color_manual(values = cb_pallete[c(8, 1, 4)], labels = c(bquote(Bias^2), "Test MSE", "Variance")) +
  #     geom_hline(yintercept = noise_sd^2, linetype = 2) + theme_bw() +
  #     theme(legend.position = "top", legend.key.spacing.x = unit(0.1, 'cm'), legend.key.size = unit(0.5, "cm")) +
  #     labs(y = "", color = "", x = "Tree Depth") +
  #     ylim(c(min(results_mse$mse) - 0.03, max(results_mse$mse) + 0.03))
  #   
  #   bias_test_var_plot
  #   
  #   
  # 
  # })
  # 
  # output$plot_lasso_p2b <- renderPlot({
  #   
  #   results <- df_bvt_calc_lasso()[[1]]
  #   noise_sd <- df_bvt_calc_lasso()[[4]]
  #   
  #   results_mse <- results %>%
  #     pivot_longer(cols = 2:5, names_to = "mse_type", values_to = "mse")
  #   
  #   bias_test_var_plot <- ggplot(data = results_mse %>% filter(mse_type != "Training")) +
  #     geom_point(aes(x = parameter_values, y = mse, color = mse_type)) +
  #     geom_line(aes(x = parameter_values, y = mse, group = mse_type, color = mse_type)) +
  #     scale_color_manual(values = cb_pallete[c(8, 1, 4)], labels = c(bquote(Bias^2), "Test MSE", "Variance")) +
  #     geom_hline(yintercept = noise_sd^2, linetype = 2) + theme_bw() +
  #     theme(legend.position = "top", legend.key.spacing.x = unit(0.1, 'cm'), legend.key.size = unit(0.5, "cm")) +
  #     labs(y = "", color = "", x = bquote(log[10](lambda))) +
  #     ylim(c(min(results_mse$mse) - 0.03, max(results_mse$mse) + 0.03))
  #   
  #   bias_test_var_plot
  #   
  #   
  # })
  # 
  # 
  # 
  # output$plot_polynomial_p2c <- renderPlot({
  #   
  #   # noise_sd <- input$epsilon          # standard deviation of epsilon
  #   
  #   results <- df_bvt_calc_polynomial()[[1]]
  # 
  #   bias_plot <- ggplot(data = results) +
  #     geom_col(aes(x = parameter_values, y = Bias_sq), fill = cb_pallete[8]) + theme_bw() +
  #     labs(y = "Estimated Squared Bias") + labs(x = "Degree")
  #   
  #   bias_plot
  #   
  # 
  # })
  # 
  # output$plot_polynomial_p2d <- renderPlot({
  #   
  #   # noise_sd <- input$epsilon          # standard deviation of epsilon
  #   
  #   results <- df_bvt_calc_polynomial()[[1]]
  #   # noise_sd <- df_bvt_calc_polynomial()[[4]]
  #   
  # 
  #   variance_plot <- ggplot(data = results) +
  #     geom_col(aes(x = parameter_values, y = Variance), fill = cb_pallete[4]) + theme_bw() +
  #     labs(y = "Estimated Variance", x = "Degree") 
  #   
  #   
  #   variance_plot
  # 
  # })
  # 
  # 
  # output$plot_spline_p2c <- renderPlot({
  #   
  #   results <- df_bvt_calc_spline()[[1]]
  #   
  #   bias_plot <- ggplot(data = results) +
  #     geom_col(aes(x = parameter_values, y = Bias_sq), fill = cb_pallete[8]) + theme_bw() +
  #     labs(y = "Estimated Squared Bias") + labs(x = "Degrees of Freedom")
  #   
  #   bias_plot
  #   
  # 
  # })
  # 
  # output$plot_spline_p2d <- renderPlot({
  #   
  #   results <- df_bvt_calc_spline()[[1]]
  #   
  #   variance_plot <- ggplot(data = results) +
  #     geom_col(aes(x = parameter_values, y = Variance), fill = cb_pallete[4]) + theme_bw() +
  #     labs(y = "Estimated Variance", x = "Degrees of Freedom") 
  #   
  #   variance_plot
  #   
  # })
  # 
  # 
  # output$plot_knn_reg_p2c <- renderPlot({
  #   
  #   results <- df_bvt_calc_knn_reg()[[1]]
  #   
  #   bias_plot <- ggplot(data = results) +
  #     geom_col(aes(x = parameter_values, y = Bias_sq), fill = cb_pallete[8]) + theme_bw() +
  #     labs(y = "Estimated Squared Bias") + labs(x = "K")
  #   
  #   bias_plot
  #   
  # 
  # })
  # 
  # output$plot_knn_reg_p2d <- renderPlot({
  #   
  #   results <- df_bvt_calc_knn_reg()[[1]]
  #   
  #   variance_plot <- ggplot(data = results) +
  #     geom_col(aes(x = parameter_values, y = Variance), fill = cb_pallete[4]) + theme_bw() +
  #     labs(y = "Estimated Variance", x = "K") 
  #   
  #   variance_plot
  #   
  # })
  # 
  # 
  # output$plot_tree_reg_p2c <- renderPlot({
  #   
  #   results <- df_bvt_calc_tree_reg()[[1]]
  #   
  #   bias_plot <- ggplot(data = results) +
  #     geom_col(aes(x = parameter_values, y = Bias_sq), fill = cb_pallete[8]) + theme_bw() +
  #     labs(y = "Estimated Squared Bias") + labs(x = "Tree Depth")
  #   
  #   bias_plot
  #   
  # 
  # })
  # 
  # output$plot_tree_reg_p2d <- renderPlot({
  #   
  #   results <- df_bvt_calc_tree_reg()[[1]]
  #   
  #   variance_plot <- ggplot(data = results) +
  #     geom_col(aes(x = parameter_values, y = Variance), fill = cb_pallete[4]) + theme_bw() +
  #     labs(y = "Estimated Variance", x = "Tree Depth") 
  #   
  #   variance_plot
  #   
  # })
  # 
  # 
  # output$plot_lasso_p2c <- renderPlot({
  #   
  #   results <- df_bvt_calc_lasso()[[1]]
  #   
  #   bias_plot <- ggplot(data = results) +
  #     geom_col(aes(x = parameter_values, y = Bias_sq), fill = cb_pallete[8]) + theme_bw() +
  #     labs(y = "Estimated Squared Bias") + labs(x = bquote(log[10](lambda)))
  #   
  #   bias_plot
  #   
  #   
  # })
  # 
  # output$plot_lasso_p2d <- renderPlot({
  #   
  #   results <- df_bvt_calc_lasso()[[1]]
  #   
  #   variance_plot <- ggplot(data = results) +
  #     geom_col(aes(x = parameter_values, y = Variance), fill = cb_pallete[4]) + theme_bw() +
  #     labs(y = "Estimated Variance", x = bquote(log[10](lambda))) 
  #   
  #   variance_plot
  #   
  # })
  
  
  
  
  
  

  ###########################
  # plots in third tab - Bias-Variance Plots
  
  poly_third_plot_data <- eventReactive(input$action_polynomial_third, {
    
    # input$action_polynomial
    
    allplots3_polynomial <- third_row_plots(model_type = "polynomial",
                    parameter_values = 1:8,
                    df_data2 = df_data_regression()[[8]],
                    x_axis_lab = "Degree")
    
    # parameter_values <- 1:8
    # x_axis_lab <- "Degree"
    # 
    # allplots3 <- subplot(df_data_regression()[[8]], df_data_regression()[[9]], df_data_regression()[[10]],
    #                      df_data_regression()[[11]], df_data_regression()[[12]], df_data_regression()[[13]],
    #                      df_data_regression()[[14]], df_data_regression()[[15]], 
    #                      nrows = 2, shareX = TRUE, titleX = TRUE) %>%
    #   layout(showlegend = FALSE, showlegend2 = TRUE, annotations = list(
    #     list(
    #       x = 0.1,
    #       y = 1.0,
    #       text = paste0(x_axis_lab, " ", parameter_values[1], " Fit"),
    #       xref = "paper",
    #       yref = "paper",
    #       xanchor = "center",
    #       yanchor = "bottom",
    #       showarrow = FALSE
    #     ),
    #     list(
    #       x = 0.4,
    #       y = 1,
    #       text = paste0(x_axis_lab, " ", parameter_values[2], " Fit"),
    #       xref = "paper",
    #       yref = "paper",
    #       xanchor = "center",
    #       yanchor = "bottom",
    #       showarrow = FALSE
    #     ),
    #     list(
    #       x = 0.65,
    #       y = 1,
    #       text = paste0(x_axis_lab, " ", parameter_values[3], " Fit"),
    #       xref = "paper",
    #       yref = "paper",
    #       xanchor = "center",
    #       yanchor = "bottom",
    #       showarrow = FALSE
    #     ),
    #     list(
    #       x = 0.85,
    #       y = 1,
    #       text = paste0(x_axis_lab, " ", parameter_values[4], " Fit"),
    #       xref = "paper",
    #       yref = "paper",
    #       xanchor = "center",
    #       yanchor = "bottom",
    #       showarrow = FALSE
    #     ),
    #     list(
    #       x = 0.1,
    #       y = 0.45,
    #       text = paste0(x_axis_lab, " ", parameter_values[5], " Fit"),
    #       xref = "paper",
    #       yref = "paper",
    #       xanchor = "center",
    #       yanchor = "bottom",
    #       showarrow = FALSE
    #     ),
    #     list(
    #       x = 0.4,
    #       y = 0.45,
    #       text = paste0(x_axis_lab, " ", parameter_values[6], " Fit"),
    #       xref = "paper",
    #       yref = "paper",
    #       xanchor = "center",
    #       yanchor = "bottom",
    #       showarrow = FALSE
    #     ),
    #     list(
    #       x = 0.65,
    #       y = 0.45,
    #       text = paste0(x_axis_lab, " ", parameter_values[7], " Fit"),
    #       xref = "paper",
    #       yref = "paper",
    #       xanchor = "center",
    #       yanchor = "bottom",
    #       showarrow = FALSE
    #     ),
    #     list(
    #       x = 0.85,
    #       y = 0.45,
    #       text = paste0(x_axis_lab, " ", parameter_values[8], " Fit"),
    #       xref = "paper",
    #       yref = "paper",
    #       xanchor = "center",
    #       yanchor = "bottom",
    #       showarrow = FALSE
    #     )))
    
      return(allplots3_polynomial)
     # return(list(df_data_regression()[[8]], df_data_regression()[[9]], df_data_regression()[[10]],
     #        df_data_regression()[[11]], df_data_regression()[[12]], df_data_regression()[[13]],
     #        df_data_regression()[[14]], df_data_regression()[[15]]))
  })
  
  spline_third_plot_data <- eventReactive(input$action_spline_third, {
    
    
    allplots3_spline <- third_row_plots(model_type = "spline",
                                 parameter_values = seq(2, 100, 13),
                                 df_data2 = df_data_regression()[[9]],
                                 x_axis_lab = "Degrees of Freedom =")
    
    return(allplots3_spline)
  })

  knn_reg_third_plot_data <- eventReactive(input$action_knn_reg_third, {
    
    
    allplots3_knn_reg <- third_row_plots(model_type = "knn_reg",
                                        parameter_values = seq(1, 15, 2),
                                        df_data2 = df_data_regression()[[10]],
                                        x_axis_lab = "K =")
    
    return(allplots3_knn_reg)
  })
  
  tree_reg_third_plot_data <- eventReactive(input$action_tree_reg_third, {
    
    
    allplots3_tree_reg <- third_row_plots(model_type = "tree_reg",
                                        parameter_values = seq(2, 16, 2),
                                        df_data2 = df_data_regression()[[11]],
                                        x_axis_lab = "Depth =")
    
    return(allplots3_tree_reg)
  })

  lasso_third_plot_data <- eventReactive(input$action_lasso_third, {
    
    
    allplots3_lasso <- third_row_plots(model_type = "lasso",
                                        parameter_values = seq(-4,3,1),
                                        df_data2 = df_data_regression()[[12]],
                                        x_axis_lab = "\u03bb = 10^")
    
    return(allplots3_lasso)
  })
  
  rf_third_plot_data <- eventReactive(input$action_rf_third, {
    
    df_training <- df_data_regression()[[1]]
    df_test <- df_data_regression()[[2]]
    df_training_xs <- df_training$x
    df_training_ys <- df_data_regression()[[13]]
    # num_training_points <- df_data_regression()[[6]]
    
    # results <- df_bvt_calc_lasso()[[1]]
    # noise_sd <- df_bvt_calc_lasso()[[4]]
    num_training_points <- nrow(df_training)
    num_test_points_reg <- nrow(df_test)
    num_rep_sets_reg <- 50                 # number of replicated datasets
    
    df_training_preds_tree <- matrix(nrow = num_training_points, ncol = num_rep_sets_reg)
    df_test_preds_tree <- matrix(nrow = num_test_points_reg, ncol = num_rep_sets_reg)
    train_mse_tree <- rep(NA, num_rep_sets_reg)
    
    # FOR AVERAGE TRAINING AND TEST MSE FROM REPLICATED DATASETS AND TEST SET
    for(j in 1:num_rep_sets_reg)
    {
      
      # fit model of d degrees of freedom to each replicated (training) set
      # get predictions (y_hat's) for each replicated (training) set and test set
      y_rep <- df_training_ys[[j]]; x_rep <- df_training_xs
      m <- rpart(y_rep ~ x_rep, method = "anova", 
                 control = rpart.control(cp = 0, xval = 0, minbucket = 1))
      y_hat_training <- predict(m, newdata = data.frame(x_rep))
      y_hat_test <- predict(m, newdata = data.frame(x_rep = df_test$x))
      
      # store predictions for each set
      df_training_preds_tree[,j] <- y_hat_training
      df_test_preds_tree[,j] <- y_hat_test
      
      # calculate training MSE for each replicated (training) dataset
      train_mse_tree[j] <- mean((y_rep-y_hat_training)^2)
      
    }
    
    
    min_y_limit <- min(sapply(df_test_preds_tree, min))
    max_y_limit <- max(sapply(df_test_preds_tree, max))
    
    df_test_preds_selected_tree <- data.frame(df_test_preds_tree)
    df_test_preds_selected_tree$mean_pred <- apply(df_test_preds_tree,1,mean)
    df_test_preds_selected_tree$expl <- df_test$x
    df_test_preds_selected_pivot_tree <- df_test_preds_selected_tree %>%
      pivot_longer(cols = starts_with("X"), names_to = "rep_data", values_to = "preds")
    
    
    df_training_preds_rf <- matrix(nrow = num_training_points, ncol = num_rep_sets_reg)
    df_test_preds_rf <- matrix(nrow = num_test_points_reg, ncol = num_rep_sets_reg)
    train_mse_rf <- rep(NA, num_rep_sets_reg)
    
    for(j in 1:num_rep_sets_reg)
    {
      
      # fit model of d degrees of freedom to each replicated (training) set
      # get predictions (y_hat's) for each replicated (training) set and test set
      y_rep <- df_training_ys[[j]]; x_rep <- df_training_xs
      
      m_rf <- randomForest(x = data.frame(x_rep), y = y_rep, xtest = data.frame(df_test$x), ytest = df_test$y_orig, #data = data.frame(y_rep = y_rep, x_rep = x_rep),
                           ntree = 50, # number of trees to grow (bootstrap samples) usually 500
                           mtry = 1, 
                           nodesize = 1)
      
      y_hat_training_rf <- m_rf$predicted
      y_hat_test_rf <- m_rf$test$predicted
      
      # store predictions for each set
      df_training_preds_rf[,j] <- y_hat_training_rf
      df_test_preds_rf[,j] <- y_hat_test_rf
      
      # calculate training MSE for each replicated (training) dataset
      train_mse_rf[j] <- mean((y_rep-y_hat_training_rf)^2)
      
    }
    
    df_test_preds_selected_rf <- data.frame(df_test_preds_rf)
    df_test_preds_selected_rf$mean_pred <- apply(df_test_preds_rf,1,mean)
    df_test_preds_selected_rf$expl <- df_test$x
    df_test_preds_selected_pivot_rf <- df_test_preds_selected_rf %>% 
      pivot_longer(cols = starts_with("X"), names_to = "rep_data", values_to = "preds")
    
    g_bias <- ggplot() + 
      geom_line(data = df_test, aes(x = x, y = fx, color = "true f(x)"), linewidth = 1) +
      # geom_line(data = df_test_preds_selected_pivot_tree, 
      #           aes(x = expl, y = preds, group = rep_data, color = "replicated fits 1"), linetype = 3) +
      geom_line(data = df_test_preds_selected_pivot_tree, 
                aes(x = expl, y = mean_pred, color = "single tree average fit"), linewidth = 1.2) +
      # geom_line(data = df_test_preds_selected_pivot_rf,
      #           aes(x = expl, y = preds, group = rep_data, color = "replicated fits 2"), linetype = 3) +
      geom_line(data = df_test_preds_selected_pivot_rf,
                aes(x = expl, y = mean_pred, color = "RF average fit"), linewidth = 1.2) +
      labs(y = "y", title = paste0("Bias Comparison")) +
      scale_colour_manual(breaks = c("true f(x)", "single tree average fit", "RF average fit"),
                          values = c("black", "orange", "darkgreen")) +
      theme_bw() +
      theme(legend.position = "bottom", legend.title = element_blank(), 
            legend.text=element_text(size=13), legend.key.size = unit(1.0, "cm"),
            legend.key = element_rect(color = NA, fill = NA)) +
      ylim(c(min_y_limit, max_y_limit)) 
    
    
    
    df_test_preds_selected_rf <- data.frame(df_test_preds_rf)
    df_test_preds_selected_rf <- df_test_preds_selected_rf %>% 
      mutate(max_pred = apply(df_test_preds_rf,1,max),
             min_pred = apply(df_test_preds_rf,1,min),
             mean_pred = apply(df_test_preds_rf,1,mean),
             expl = df_test$x)
    df_test_preds_selected_tree <- data.frame(df_test_preds_tree)
    df_test_preds_selected_tree <- df_test_preds_selected_tree %>% 
      mutate(max_pred = apply(df_test_preds_tree,1,max),
             min_pred = apply(df_test_preds_tree,1,min),
             mean_pred = apply(df_test_preds_tree,1,mean),
             expl = df_test$x)
    g_variance <- ggplot() + 
      geom_line(data = df_test, aes(x = x, y = fx, color = "true f(x)"), linewidth = 1.5) +
      geom_linerange(data = df_test_preds_selected_tree,
                     aes(x = expl, y = mean_pred, ymin = min_pred, ymax = max_pred, color = "single tree fit"), linewidth = 0.3) +
      geom_linerange(data = df_test_preds_selected_rf,
                     aes(x = expl, y = mean_pred, ymin = min_pred, ymax = max_pred, color = "RF fit"), linewidth = 0.3) +
      labs(y = "y", title = paste0("Variance Comparison")) +
      scale_colour_manual(breaks = c("true f(x)", "single tree fit", "RF fit"),
                          values = c("black", "orange", "darkgreen")) +
      theme_bw() +
      theme(legend.position = "bottom", legend.title = element_blank(), 
            legend.text=element_text(size=13), legend.key.size = unit(1.0, "cm"),
            legend.key = element_rect(color = NA, fill = NA)) +
      ylim(c(min_y_limit, max_y_limit)) 
    
    
    return(list(g_bias, g_variance))
  })
  
  
  
  output$plot_polynomial_p3 <- renderPlotly({
    
    # df_data_regression()$allplots3_polynomial
    
    # third_row_plots_m(model_type = "polynomial", 
    #                 df_data1 = df_data_regression(), 
    #                 df_data2 = df_bvt_calc_polynomial(), 
    #                 x_axis_lab = "Degree ")
    # observeEvent(input$action_polynomial_third, {
    
    
    # })
    
    
    # req(input$action_polynomial_third)
    # third_row_plots(model_type = "polynomial",
    #                 parameter_values = 1:8,
    #                 df_data2 = df_data_regression()[[8]],
    #                 x_axis_lab = "Degree")
    
    # third_row_plots(model_type = "polynomial",
    #                 parameter_values = 1:8,
    #                 df_data2 = poly_third_plot_data(),
    #                 x_axis_lab = "Degree")
    
    # req(plot_visible())
    
    poly_third_plot_data()
    
  }) %>% bindCache(poly_third_plot_data())
  # %>% bindCache(poly_third_plot_data())
  # %>% 
  #   bindCache(df_data_regression(), df_bvt_calc_polynomial())
  # , cacheKeyExpr = { list(input$dataset_polynomial, input$num_points_polynomial, input$epsilon_polynomial, df_data_regression(), df_bvt_calc_polynomial()) })
  
  output$plot_spline_p3 <- renderPlotly({
    
    # df_data_regression()$allplots3_spline
    
    # third_row_plots_m(model_type = "spline", 
    #                 df_data1 = df_data_regression(), 
    #                 df_data2 = df_bvt_calc_spline(), 
    #                 x_axis_lab = "Degrees of Freedom = ")
    
    # third_row_plots(model_type = "spline",
    #                 df_data1 = df_data_regression()[[4]],
    #                 df_data2 = df_data_regression()[[9]],
    #                 x_axis_lab = "Degrees of Freedom =")
    
    spline_third_plot_data()
    
  }) %>% bindCache(spline_third_plot_data())
    # bindCache(df_data_regression(), df_bvt_calc_spline())
  # , cacheKeyExpr = { list(input$dataset_polynomial, input$num_points_polynomial, input$epsilon_polynomial, df_data_regression(), df_bvt_calc_spline()) }) 
  
  output$plot_knn_reg_p3 <- renderPlotly({
    
    # df_data_regression()$allplots3_knn_reg
    
    # third_row_plots_m(model_type = "knn_reg", 
    #                 df_data1 = df_data_regression(), 
    #                 df_data2 = df_bvt_calc_knn_reg(), 
    #                 x_axis_lab = "K = ")
    
    # third_row_plots(model_type = "knn_reg",
    #                 df_data1 = df_data_regression()[[5]],
    #                 df_data2 = df_data_regression()[[10]],
    #                 x_axis_lab = "K =")
    # 
    
    knn_reg_third_plot_data()
    
  }) %>% bindCache(knn_reg_third_plot_data())
    # bindCache(df_data_regression(), df_bvt_calc_knn_reg())
  # , cacheKeyExpr = { list(input$dataset_polynomial, input$num_points_polynomial, input$epsilon_polynomial, df_data_regression(), df_bvt_calc_knn_reg()) })
  
  output$plot_tree_reg_p3 <- renderPlotly({
    
    # df_data_regression()$allplots3_tree_reg
    
    # third_row_plots_m(model_type = "tree_reg", 
    #                 df_data1 = df_data_regression(), 
    #                 df_data2 = df_bvt_calc_tree_reg(), 
    #                 x_axis_lab = "Depth = ")
    
    # third_row_plots(model_type = "tree_reg",
    #                 df_data1 = df_data_regression()[[6]],
    #                 df_data2 = df_data_regression()[[11]],
    #                 x_axis_lab = "Depth =")
    
    tree_reg_third_plot_data()
    
  }) %>% bindCache(tree_reg_third_plot_data())
    # bindCache(df_data_regression(), df_bvt_calc_tree_reg())
  # , cacheKeyExpr = { list(input$dataset_polynomial, input$num_points_polynomial, input$epsilon_polynomial, df_data_regression(), df_bvt_calc_tree_reg()) }) 
  
  output$plot_lasso_p3 <- renderPlotly({
    
    # df_data_regression()$allplots3_lasso
    
    # third_row_plots_m(model_type = "lasso", 
    #                 df_data1 = df_data_regression(), 
    #                 df_data2 = df_bvt_calc_lasso(), 
    #                 x_axis_lab = "\u03bb = 10^")
    
    # third_row_plots(model_type = "lasso",
    #                 df_data1 = df_data_regression()[[7]],
    #                 df_data2 = df_data_regression()[[12]],
    #                 x_axis_lab = "\u03bb = 10^")
    
    lasso_third_plot_data()
    
  }) %>% bindCache(lasso_third_plot_data())
    # bindCache(df_data_regression(), df_bvt_calc_lasso())
  # , cacheKeyExpr = { list(input$dataset_polynomial, input$num_points_polynomial, input$epsilon_polynomial, df_data_regression(), df_bvt_calc_lasso()) }) 
  
  output$plot_rf_p3 <- renderCachedPlot({
    
    rf_third_plot_data()[[1]] + rf_third_plot_data()[[2]]
    # g_bias + g_variance
    
    
  }, cacheKeyExpr = { list(rf_third_plot_data()) })   
  
  ###########################
  # old code for plots, replaced by a function and code above
  
  # output$plot_polynomial_p3 <- renderPlot({
  # 
  #   # if(input$bvgraphs == TRUE){
  # 
  #   # degrees_of_freedom <- df()[[3]]
  #   df_test <- df_data_regression()[[2]]
  #   parameter_values <- df_bvt_calc_polynomial()[[2]]
  #   df_test_preds_list <- df_bvt_calc_polynomial()[[3]]
  #   min_y_limit <- min(sapply(df_test_preds_list, min))
  #   max_y_limit <- max(sapply(df_test_preds_list, max))
  # 
  # 
  #   replicated_datasets_graphs <- list()
  #   for(count in seq_along(parameter_values))
  #   {
  #     # df_test_preds_selected <- data.frame(df_test_preds_list[[count]][,sample_reps])
  #     df_test_preds_selected <- data.frame(df_test_preds_list[[count]])
  #     df_test_preds_selected$mean_pred <- apply(df_test_preds_list[[count]],1,mean)
  #     df_test_preds_selected$expl <- df_test$x
  #     df_test_preds_selected_pivot <- df_test_preds_selected %>%
  #       pivot_longer(cols = starts_with("X"), names_to = "rep_data", values_to = "preds")
  # 
  #     replicated_datasets_graphs[[count]] <- ggplot() +
  #       geom_line(data = df_test, aes(x = x, y = fx, color = "true f(x)"), linewidth = 1.5) +  # MAYBE CHANGE LINEWIDTH AND LAYER THIS AFTER THE AVERAGE FIT
  #       geom_line(data = df_test_preds_selected_pivot,
  #                 aes(x = expl, y = preds, group = rep_data, color = "replicated fits"), linetype = 3) +
  #       geom_line(data = df_test_preds_selected_pivot,
  #                 aes(x = expl, y = mean_pred, color = "average fit")) +    # ADD LINEWIDTH = 1.3
  #       # geom_line(data = df_test, aes(x = x, y = fx, color = "true f(x)"), linewidth = 0.8) +
  #       labs(y = "y", title = paste0("Degree ", parameter_values[count], " Fit")) +
  #       scale_colour_manual(breaks = c("true f(x)", "replicated fits", "average fit"),
  #                           values = c("black", "orange", "red")) +    # colorblind_pal()(7)[-c(3:6)]
  #       theme_bw() +
  #       theme(legend.position = "top", legend.title = element_blank(),
  #             legend.text=element_text(size=13), legend.key.size = unit(1, "cm"),
  #             legend.key.spacing.x = unit(2.0, "cm"), legend.key = element_rect(color = NA, fill = NA)) +
  #       ylim(c(min_y_limit, max_y_limit))
  # 
  #     # NEED TO WORK ON THE LEGEND SPACING
  # 
  #   }
  # 
  # 
  #   (replicated_datasets_graphs[[1]] | replicated_datasets_graphs[[2]] | replicated_datasets_graphs[[3]] | replicated_datasets_graphs[[4]]) /
  #     (replicated_datasets_graphs[[5]] | replicated_datasets_graphs[[6]] | replicated_datasets_graphs[[7]] | replicated_datasets_graphs[[8]]) + plot_layout(guides = "collect") & theme(legend.position = "bottom")
  # 
  # 
  #   # }
  # 
  # 
  # })
  # 
  # output$plot_spline_p3 <- renderPlot({
  #   
  #   df_test <- df_data_regression()[[2]]
  #   parameter_values <- df_bvt_calc_spline()[[2]]
  #   df_test_preds_list <- df_bvt_calc_spline()[[3]]
  #   min_y_limit <- min(sapply(df_test_preds_list, min))
  #   max_y_limit <- max(sapply(df_test_preds_list, max))
  #   
  #   replicated_datasets_graphs <- list()
  #   for(count in seq_along(parameter_values))
  #   {
  #     # df_test_preds_selected <- data.frame(df_test_preds_list[[count]][,sample_reps])
  #     df_test_preds_selected <- data.frame(df_test_preds_list[[count]])
  #     df_test_preds_selected$mean_pred <- apply(df_test_preds_list[[count]],1,mean)
  #     df_test_preds_selected$expl <- df_test$x
  #     df_test_preds_selected_pivot <- df_test_preds_selected %>% 
  #       pivot_longer(cols = starts_with("X"), names_to = "rep_data", values_to = "preds")
  #     
  #     replicated_datasets_graphs[[count]] <- ggplot() + 
  #       geom_line(data = df_test, aes(x = x, y = fx, color = "true f(x)"), linewidth = 1.5) +
  #       geom_line(data = df_test_preds_selected_pivot, 
  #                 aes(x = expl, y = preds, group = rep_data, color = "replicated fits"), linetype = 3) +
  #       geom_line(data = df_test_preds_selected_pivot, 
  #                 aes(x = expl, y = mean_pred, color = "average fit")) +
  #       labs(y = "y", title = paste0("Degrees of Freedom = ", parameter_values[count], " Fit")) +
  #       scale_colour_manual(breaks = c("true f(x)", "replicated fits", "average fit"),
  #                           values = c("black", "orange", "red")) +
  #       theme_bw() +
  #       theme(legend.position = "top", legend.title = element_blank(), 
  #             legend.text=element_text(size=13), legend.key.size = unit(1.0, "cm"),
  #             legend.key.spacing.x = unit(2.0, "cm"), legend.key = element_rect(color = NA, fill = NA)) +
  #       ylim(c(min_y_limit, max_y_limit)) 
  #   }
  #   
  # 
  #   (replicated_datasets_graphs[[1]] | replicated_datasets_graphs[[2]] | replicated_datasets_graphs[[3]] | replicated_datasets_graphs[[4]]) /
  #     (replicated_datasets_graphs[[5]] | replicated_datasets_graphs[[6]] | replicated_datasets_graphs[[7]] | replicated_datasets_graphs[[8]]) + plot_layout(guides = "collect") & theme(legend.position = "bottom")
  #   
  #   
  # })
  # 
  # output$plot_knn_reg_p3 <- renderPlot({
  #   
  #   df_test <- df_data_regression()[[2]]
  #   parameter_values <- df_bvt_calc_knn_reg()[[2]]
  #   df_test_preds_list <- df_bvt_calc_knn_reg()[[3]]
  #   min_y_limit <- min(sapply(df_test_preds_list, min))
  #   max_y_limit <- max(sapply(df_test_preds_list, max))
  #   
  #   replicated_datasets_graphs <- list()
  #   for(count in seq_along(parameter_values))
  #   {
  #     # df_test_preds_selected <- data.frame(df_test_preds_list[[count]][,sample_reps])
  #     df_test_preds_selected <- data.frame(df_test_preds_list[[count]])
  #     df_test_preds_selected$mean_pred <- apply(df_test_preds_list[[count]],1,mean)
  #     df_test_preds_selected$expl <- df_test$x
  #     df_test_preds_selected_pivot <- df_test_preds_selected %>% 
  #       pivot_longer(cols = starts_with("X"), names_to = "rep_data", values_to = "preds")
  #     
  #     replicated_datasets_graphs[[count]] <- ggplot() + 
  #       geom_line(data = df_test, aes(x = x, y = fx, color = "true f(x)"), linewidth = 1.5) +
  #       geom_line(data = df_test_preds_selected_pivot, 
  #                 aes(x = expl, y = preds, group = rep_data, color = "replicated fits"), linetype = 3) +
  #       geom_line(data = df_test_preds_selected_pivot, 
  #                 aes(x = expl, y = mean_pred, color = "average fit")) +
  #       # geom_line(data = df_test, aes(x = x, y = fx, color = "true f(x)"), linewidth = 0.8) +
  #       labs(y = "y", title = paste0("K = ", parameter_values[count], " Fit")) +
  #       scale_colour_manual(breaks = c("true f(x)", "replicated fits", "average fit"),
  #                           values = c("black", "orange", "red")) +
  #       theme_bw() +
  #       theme(legend.position = "top", legend.title = element_blank(), 
  #             legend.text=element_text(size=13), legend.key.size = unit(1.0, "cm"),
  #             legend.key.spacing.x = unit(2.0, "cm"), legend.key = element_rect(color = NA, fill = NA)) +
  #       ylim(c(min_y_limit, max_y_limit))
  #     
  #   }
  #   
  #   
  #   (replicated_datasets_graphs[[1]] | replicated_datasets_graphs[[2]] | replicated_datasets_graphs[[3]] | replicated_datasets_graphs[[4]]) /
  #     (replicated_datasets_graphs[[5]] | replicated_datasets_graphs[[6]] | replicated_datasets_graphs[[7]] | replicated_datasets_graphs[[8]]) + plot_layout(guides = "collect") & theme(legend.position = "bottom")
  #   
  #   
  # })
  # 
  # output$plot_tree_reg_p3 <- renderPlot({
  #   
  #   df_test <- df_data_regression()[[2]]
  #   parameter_values <- df_bvt_calc_tree_reg()[[2]]
  #   df_test_preds_list <- df_bvt_calc_tree_reg()[[3]]
  #   min_y_limit <- min(sapply(df_test_preds_list, min))
  #   max_y_limit <- max(sapply(df_test_preds_list, max))
  #   
  #   replicated_datasets_graphs <- list()
  #   for(count in seq_along(parameter_values))
  #   {
  #     # df_test_preds_selected <- data.frame(df_test_preds_list[[count]][,sample_reps])
  #     df_test_preds_selected <- data.frame(df_test_preds_list[[count]])
  #     df_test_preds_selected$mean_pred <- apply(df_test_preds_list[[count]],1,mean)
  #     df_test_preds_selected$expl <- df_test$x
  #     df_test_preds_selected_pivot <- df_test_preds_selected %>% 
  #       pivot_longer(cols = starts_with("X"), names_to = "rep_data", values_to = "preds")
  #     
  #     replicated_datasets_graphs[[count]] <- ggplot() + 
  #       geom_line(data = df_test, aes(x = x, y = fx, color = "true f(x)"), linewidth = 1.5) +
  #       geom_line(data = df_test_preds_selected_pivot, 
  #                 aes(x = expl, y = preds, group = rep_data, color = "replicated fits"), linetype = 3) +
  #       geom_line(data = df_test_preds_selected_pivot, 
  #                 aes(x = expl, y = mean_pred, color = "average fit")) +
  #       labs(y = "y", title = paste0("Depth = ", parameter_values[count], " Fit")) +
  #       scale_colour_manual(breaks = c("true f(x)", "replicated fits", "average fit"),
  #                           values = c("black", "orange", "red")) +
  #       theme_bw() +
  #       theme(legend.position = "top", legend.title = element_blank(), 
  #             legend.text=element_text(size=13), legend.key.size = unit(1.0, "cm"),
  #             legend.key.spacing.x = unit(2.0, "cm"), legend.key = element_rect(color = NA, fill = NA)) +
  #       ylim(c(min_y_limit, max_y_limit)) 
  #   }
  #   
  # 
  #   (replicated_datasets_graphs[[1]] | replicated_datasets_graphs[[2]] | replicated_datasets_graphs[[3]] | replicated_datasets_graphs[[4]]) /
  #     (replicated_datasets_graphs[[5]] | replicated_datasets_graphs[[6]] | replicated_datasets_graphs[[7]] | replicated_datasets_graphs[[8]]) + plot_layout(guides = "collect") & theme(legend.position = "bottom")
  #   
  #   
  # })
  # 
  # output$plot_lasso_p3a <- renderPlot({
  #   
  #   df_test <- df_data_regression()[[2]]
  #   parameter_values <- df_bvt_calc_lasso()[[2]]
  #   df_test_preds_list <- df_bvt_calc_lasso()[[3]]
  #   # min_y_limit <- min(sapply(df_test_preds_list, min))
  #   # max_y_limit <- max(sapply(df_test_preds_list, max))
  #   min_y_limit <- min(df_test$y_orig)
  #   max_y_limit <- max(max(df_test$y_orig))
  #   
  #   replicated_datasets_graphs <- list()
  #   for(count in seq_along(parameter_values))
  #   {
  #     # df_test_preds_selected <- data.frame(df_test_preds_list[[count]][,sample_reps])
  #     df_test_preds_selected <- data.frame(df_test_preds_list[[count]])
  #     df_test_preds_selected$mean_pred <- apply(df_test_preds_list[[count]],1,mean)
  #     df_test_preds_selected$expl <- df_test$x
  #     df_test_preds_selected_pivot <- df_test_preds_selected %>% 
  #       pivot_longer(cols = starts_with("X"), names_to = "rep_data", values_to = "preds")
  #     
  #     replicated_datasets_graphs[[count]] <- ggplot() + 
  #       geom_line(data = df_test, aes(x = x, y = fx, color = "true f(x)"), linewidth = 1.5) +
  #       geom_line(data = df_test_preds_selected_pivot, 
  #                 aes(x = expl, y = preds, group = rep_data, color = "replicated fits"), linetype = 3) +
  #       geom_line(data = df_test_preds_selected_pivot, 
  #                 aes(x = expl, y = mean_pred, color = "average fit")) +
  #       labs(y = "y", title = paste0("\u03bb"," = 10^", log10(parameter_values[count]), " Fit")) +
  #       scale_colour_manual(breaks = c("true f(x)", "replicated fits", "average fit"),
  #                           values = c("black", "orange", "red")) +
  #       theme_bw() +
  #       theme(legend.position = "top", legend.title = element_blank(), 
  #             legend.text=element_text(size=13), legend.key.size = unit(1.0, "cm"),
  #             legend.key.spacing.x = unit(2.0, "cm"), legend.key = element_rect(color = NA, fill = NA)) +
  #       ylim(c(min_y_limit, max_y_limit)) 
  #   }
  #   
  # 
  #   (replicated_datasets_graphs[[1]] | replicated_datasets_graphs[[2]] | replicated_datasets_graphs[[3]] | replicated_datasets_graphs[[4]]) /
  #     (replicated_datasets_graphs[[5]] | replicated_datasets_graphs[[6]] | replicated_datasets_graphs[[7]] | replicated_datasets_graphs[[8]]) + plot_layout(guides = "collect") & theme(legend.position = "bottom")
  #   
  #   
  # })
  # 
  # # output$plot_lasso_p3b <- renderPlot({
  # #   
  # #   parameter_values <- df_bvt_calc_lasso()[[2]]
  # #   model_coefficients <- df_bvt_calc_lasso()[[5]]
  # #   poly_degree <- 5
  # #   
  # #   coefs <- plyr::ldply(model_coefficients, data.frame)
  # #   colnames(coefs) <- paste0("beta", 1:poly_degree)
  # #   coefs$lambda <- rep(log10(parameter_values), each = num_rep_sets_reg)
  # #   coefs2 <- pivot_longer(coefs, cols = starts_with("b"), names_to = "coef", values_to = "coef_values")
  # #   coefs2$coef <- factor(coefs2$coef, levels = paste0("beta", 1:poly_degree), 
  # #                         labels = paste0("beta[", 1:poly_degree, "]"))
  # #   
  # #   ggplot(data = coefs2, aes(x = factor(lambda), y = coef_values)) + 
  # #     geom_boxplot() + 
  # #     facet_wrap(~ coef, scales = "free_y", nrow = 2, ncol = 5, labeller = label_parsed) +
  # #     labs(y = "Coefficient Estimates", x = bquote(log[10](lambda)))    
  # # })
  # # 
  
 
  
  

  

  
  
}


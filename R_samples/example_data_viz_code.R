# Title: Potentially helpful data viz code
# Author: Mary Lofton
# Date: 27FEB24

# Purpose: provide example data visualization code for VERA forecast target data
# to CMDA capstone students

# Install/load packages
install.packages("pacman")

# this function checks to see if packages are installed before loading and if not,
# installs them
pacman::p_load(tidyverse, lubridate)

# Access data
url <- "https://renc.osn.xsede.org/bio230121-bucket01/vera4cast/targets/project_id=vera4cast/duration=P1D/daily-insitu-targets.csv.gz"
targets <- read_csv(url, show_col_types = FALSE)

# Filter to bloom binary target
plotdata <- targets %>%
  filter(depth_m %in% c(1.6, 1.5) & variable == "Bloom_binary_mean" & duration == "P1D")

# Plot both reservoirs
ggplot(data = plotdata, aes(x = datetime, y = observation))+ 
  geom_point()+
  facet_grid(rows = vars(site_id), scales = "free_y")+
  theme_bw()+
  ggtitle("Both")

# Plot each reservoir separately
plotdata1 <- targets %>%
  filter(site_id == "bvre" & depth_m == 1.5 & variable == "Bloom_binary_mean" & duration == "P1D")

ggplot(data = plotdata1, aes(x = datetime, y = observation))+
  geom_point()+
  theme_bw()+
  ggtitle("Beaverdam Reservoir")

plotdata2 <- targets %>%
  filter(site_id == "fcre" & depth_m == 1.6 & variable == "Bloom_binary_mean" & duration == "P1D")

ggplot(data = plotdata, aes(x = datetime, y = observation))+
  geom_point()+
  theme_bw()+
  ggtitle("Falling Creek Reservoir")

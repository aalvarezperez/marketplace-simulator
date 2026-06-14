require(dplyr)
require(readr)
require(ggplot2)
source("colortools.R")
source("vintalisation.R")

df_users <- read_csv("df_users.csv")
df_users_effect <- read_csv("df_users_effect.csv")

df_perm_noeffect <- read_csv("permutations_no_effect.csv")
df_perm_effect <- read_csv("permutations_effect.csv")


df_listings <- read_csv("listings_views.csv")

df_listings <- df_listings %>%
  group_by(entity) %>%
  summarise(
    n_views = sum(views)
  )
  

df_listings %>%
  ggplot(aes(n_views)) + 
  geom_histogram(bins=100, fill='#FF8759') + 
  theme_adevinta() + 
  labs(
    y='Listings',
    x='Number of Views'
  )
 
# make_plot('listings-views.png', height = 6, unit='in')

empirical_effect = 0.7490025651126628
empirical_no_effect = -0.06322976683527048
actual_effect <- .5

text_col <- "#2900d2"

df_users %>%
  group_by(variant) %>%
  summarise(
    n = n(),
    mean(transactions)
  )

df_users_effect %>%
  group_by(variant) %>%
  summarise(
    n = n(),
    mean(transactions)
  )


# quality and engagement
tibble(x = rgamma(100000, .2)) %>%
  ggplot(aes(x)) + 
  geom_histogram(bins=100,
                 fill='#b6aaf9') + 
  theme_adevinta() + 
  labs(
    y='Users\ncount',
    x='Engagement score'
  )

make_plot('engagement-score.png', height = 6, unit='in')

tibble(x = rgamma(100000, .2)) %>%
  ggplot(aes(x)) + 
  geom_histogram(bins=100,
                 fill='#ecb3c7') + 
  theme_adevinta() + 
  labs(
    y='Listing\ncount',
    x='Quality score'
  )


df_users %>%
  filter(!is.na(variant)) %>%
  ggplot(aes(visits)) + 
  geom_histogram(position = 'dodge', bins = 100, fill='#98cd94') +
  theme_adevinta() +
  scale_x_continuous(breaks=seq(0, 10, 1)) + 
  labs(
    y='Users',
    x='Number of visits'
  )

make_plot('visits.png', height = 6, unit='in')


df_users_effect %>%
  filter(!is.na(variant)) %>%
  ggplot(aes(transactions)) + 
  geom_histogram(position = 'dodge', bins = 100, fill='#b6aaf9') +
  theme_adevinta() +
  labs(
    y='Users',
    x='Number of transactions'
  )

make_plot('transactions.png', height = 6, unit='in')

df_users_effect %>%
  filter(!is.na(variant)) %>%
  ggplot(aes(listings_placed)) + 
  geom_histogram(position = 'dodge', bins = 100, fill='#91b495') +
  theme_adevinta() +
  labs(
    y='Users',
    x='Number of listings placed'
  )

make_plot('listings-placed.png', height = 6, unit='in')


df_perm_noeffect %>%
  ggplot(aes(`0`)) + 
  geom_histogram(position = 'dodge', bins = 70, fill='#E85D7B') + 
  geom_vline(xintercept = empirical_no_effect, lty='dashed', linewidth=1, color="#2900d2") +
  geom_vline(xintercept = 0, lty='dashed', linewidth=1, color="grey") + 
  annotate('text', x = empirical_no_effect-.05, y=100,
           label='Estimated\nRel.Uplift',
           color="#2900d2") +
  scale_x_continuous(labels=scales::percent) +
  theme_adevinta() +
  labs(
    y='Permutations',
    x='Relative Uplift'
  )
  
make_plot('permutations-no-effect.png', height = 6, unit='in')

df_perm_effect %>%
  ggplot(aes(`0`)) + 
  geom_histogram(position = 'dodge', bins = 70, fill='#5977FF') + 
  geom_vline(xintercept = empirical_effect, lty='dashed', linewidth=1, color="#2900d2") +
  geom_vline(xintercept = .5, lty='dashed', linewidth=1, color="grey") + 
  annotate('text', x = empirical_effect-.05, y=125,
           label='Estimated\nRel.Uplift',
           color="#2900d2") +
  annotate('text', x = actual_effect-.05, y=125,
           label='Actual\nRel.Uplift',
           color="#2900d2") +
  scale_x_continuous(labels=scales::percent) +
  theme_adevinta() +
  labs(
    y='Permutations',
    x='Relative Uplift'
  )

make_plot('permutations-effect.png', height = 6, unit='in')


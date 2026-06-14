


colors_vec <- c(
  # template colors
  `sky_blue`   = "#bde2ea",
  `jungle`     = "#91b495",
  `lavender`   = "#b6aaf9",
  `orange`     = "#ffd684",
  `lime`       = "#eef2a6",
  `pink`       = "#ecb3c7",
  `purple`     = "#6e0ad6",
  
  skyblue_blue_1 = "#bde2ea",
  skyblue_viole_2 = "#8959FF",
  skyblue_red_3 = "#E85D7B",
  skyblue_oran_4 = "#FFB459",
  skyblue_yello_5 = "#F5F156",
  
  lavender_lave_1 = "#b6aaf9",
  lavender_lila_2 = "#DC59FF",
  lavender_red_3 = "#E85D7B",
  lavender_oran_4 = "#FF8759",
  lavender_yell_5 = "#F5B756",
  
  jungle_gree_1 = "#91b495",
  jungle_oliv_2 = "#FFD359",
  jungle_salm_3 = "#E85D7B",
  jungle_pink_4 = "#5977FF",
  jungle_lila_5 = "#56F576",
  
  orange_oran_1 = "#ffd684",
  orange_red_2 = "#F03A30",
  orange_purpl_3 = "#7D36D9",
  orange_blue_4 = "#30C0F0",
  orange_gree_5 = "#2EE62F",
  
  # grey and pastel
  orange_blue_1 = "#e79041",
  orange_blue_2 = "#bbb559",
  orange_blue_3 = "#98cd94",
  orange_blue_4 = "#96dbcc",
  orange_blue_5 = "#bde2ea",
  
  adobe_1_skyb_1 = "#A3CDD9",
  adobe_1_beige_2 = "#FFFCE6",
  adobe_1_yello_3 = "#F2CC39",
  adobe_1_blue_4 = "#506AD4",
  adobe_1_grey_5 = "#C2B8AD",
  
  light_grey = "#cccccc",
  dark_grey  = "#8c8c8c",
  `red`        = "#d11141",
  `green`      = "#00b159",
  
  marktplaats1_oran_1 = "#EDA566",
  marktplaats1_green_2 = "#A4E831",
  marktplaats1_oran_3 = "#E87A1A",
  marktplaats1_blue_4 = "#02C0E8",
  marktplaats1_purpl_5 = "#B80EE8",
  
  gumtree_viole_1 = "#3C3241",
  gumtree_oran_2 = "#E37327",
  gumtree_purpl_3 = "#9D10E3",
  gumtree_green_4 = "#B9E327",
  gumtree_blue_5 = "#05D5E3",
  
  kijiji_purpl_1 = "#373373",
  kijiji_red_2 = "#E64839",
  kijiji_blue_3 = "#2F22E6",
  kijiji_yello_4 = "#E6C30B",
  kijiji_green_5 = "#17E673",
  
  colorblind1_pink = "#E69F00",
  colorblind1_red = "#56B4E9",
  colorblind1_blue = "#009E73",
  colorblind1_yellow = "#F0E442",
  colorblind1_green = "#0072B2",
  colorblind1_skyblue = "#D55E00",
  colorblind1_orange = "#CC79A7",
  
  coolors1_Charcoal = "#264653",
  coolors1_Charcoal_Persian_Green = "#2a9d8f",
  coolors1_Charcoal_Orange_Yellow_Crayola = "#e9c46a",
  coolors1_Charcoal_Sandy_Brown = "#f4a261",
  coolors1_Charcoal_Burnt_Sienna = "#e76f51",
  
  
  coolors2_Imperial_Red = "#e63946",
  coolors2_Honeydew = "#f1faee",
  coolors2_Powder_Blue = "#a8dadc",
  coolors2_Celadon_Blue = "#457b9d",
  coolors2_Prussian_Blue = "#1d3557"
)


get_color <- function(...) {
  cols <- c(...)
  
  if (is.null(cols))
    return (colors_vec)
  
  colors_vec[cols]
}

palettes_list <- list(
  main  = get_color("sky_blue", "jungle", "lavender", "orange", "lime", "pink", "purple"),
  skyblue  = get_color(names(colors_vec)[grepl("skyblue", names(colors_vec))]),
  orange = get_color(names(colors_vec)[grepl("orange", names(colors_vec))]),
  jungle = get_color(names(colors_vec)[grepl("jungle", names(colors_vec))]),
  lavender  = get_color(names(colors_vec)[grepl("lavender", names(colors_vec))]),
  beach = get_color(names(colors_vec)[grepl("orange", names(colors_vec))]),
  adobe1 = get_color(names(colors_vec)[grepl("adobe_1", names(colors_vec))]),
  dark2 = RColorBrewer::brewer.pal(8, "Dark2"),
  pastel1 = RColorBrewer::brewer.pal(8, "Pastel1"),
  pastel2 = RColorBrewer::brewer.pal(8, "Pastel2"),
  marktplaats = get_color(names(colors_vec)[grepl("marktplaats1", names(colors_vec))]),
  gumtree = get_color(names(colors_vec)[grepl("gumtree", names(colors_vec))]),
  kijiji = get_color(names(colors_vec)[grepl("kijiji", names(colors_vec))]),
  colorblind = get_color(names(colors_vec)[grepl("colorblind1", names(colors_vec))]),
  coolors1 = get_color(names(colors_vec)[grepl("coolors1", names(colors_vec))]),
  coolors2 = get_color(names(colors_vec)[grepl("coolors2", names(colors_vec))])
)



#'search over color wheel in a way to obtain a palette given a base color.
#' @param base_color starting hex value to generate the palette
#' @param how In which way to search over the colorwheel: square, tetratic, analogous, complementary
#' @param plot Boolean for plotting returned values
search_color_wheel <- function(base_color, how = "tetradic", plot = TRUE){
  
  regex = "^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$"
  if(regexpr(text = base_color, pattern = regex) != 1) stop("enter a valid hex value for the base color")
  if(is.null(how)) stop("Looking for colors? select analogous, complementary, tetradic or square search over the color wheel.")
  
  switch(how,
         analogous = analogous(color = base_color, plot = plot),
         complementary = complementary(color = base_color, plot = plot),
         tetradic = tetradic(color = base_color, plot = plot),
         square = square(color = base_color, plot = plot)
  )
}


get_palette <- function(palette = "main", reverse = FALSE, ...) {
  if(!palette %in% names(palettes_list)) stop("Provided palette name does not exist. See available palettes with show_palettes().")
  pal <- palettes_list[[palette]]
  if(reverse) pal <- rev(pal)
  grDevices::colorRampPalette(pal, ...)
}


generate_palette <- function(base_color, how, reverse = FALSE, ...){
  pal <- search_color_wheel(base_color, how)
  if(reverse) pal <- rev(pal)
  grDevices::colorRampPalette(pal, ...)
}



#' This function allows the user to extract built-in palettes and check out the color values.
#' @param palette A string with the name of the built-in palette. If NULL, the default, then all palette names will be printed.
#' @param reverse Boolean to reverse the color order within the palette.
#' @param plot Boolean to plot the returned hex values.
#' @return A vector of hex values that make up the palette
#' @export
show_palette <- function(palette = NULL, reverse = FALSE, plot = TRUE) {
  if(is.null(palette)){
    return(names(palettes_list))
  } else if(!(palette %in% names(palettes_list))){
    stop("Provided palette name does not exist. See available palettes with show_palettes(palette = NULL).")
  }
  
  pal <- palettes_list[[palette]]
  
  if(reverse) pal <- rev(pal)
  
  if(plot) scales::show_col(pal)
  
  pal
}





#' Adevinta's color palette
#' this function lets you manage colors ggplot, like ggplot2::scale_color_manual()
#' @param palette A string with the name of the built-in palette. If NULL, the default, then all palette names will be printed.
#' @param discrete Boolean for the nature of the mapped aesthetic. TRUE is default.
#' @param reverse Boolean to reverse the color order within the palette.
#' @param ... other parameters that the user wants to pass on to ggplot2::discrete_scale(.)
#' @export
scale_color_adevinta <- function(palette = "main", discrete = TRUE, reverse = FALSE, ...) {
  pal <- get_palette(palette = palette, reverse = reverse)
  
  if (discrete) {
    discrete_scale("colour", paste0("adevinta_", palette), palette = pal, ...)
  } else {
    scale_color_gradientn(colours = pal(256), ...)
  }
}



#' Adevinta's color palette
#'
#' this function lets you manage colors ggplot, like ggplot2::scale_color_manual()
#' as opposed to scale_color_adevinta(), which takes a pre-defined palette name,
#' this function takes a hex value and a 'way' to search around the color wheel to generate a palette
#' @param base_color starting hex value to generate the palette
#' @param how in which way to search over the colorwheel: square, tetratic, analogous, complementary
#' @param discrete Boolean for the nature of the mapped aesthetic. TRUE is default.
#' @param reverse Boolean to reverse the color order within the palette.
#' @param ... other parameters that the user wants to pass on to ggplot2::discrete_scale(.)
#' @export
scale_color_adevinta_manual <- function(base_color, how, discrete = TRUE, reverse = FALSE, ...) {
  pal <- generate_palette(base_color, how)
  
  if (discrete) {
    discrete_scale("colour", paste0(base_color, "_", how, "_based_palette"), palette = pal, ...)
  } else {
    scale_color_gradientn(colours = pal(256), ...)
  }
}


#' Adevinta's color palette
#'
#' this function lets you manage colors ggplot, like ggplot2::scale_color_manual()
#' @param palette A string with the name of the built-in palette. If NULL, the default, then all palette names will be printed.
#' @param discrete Boolean for the nature of the mapped aesthetic. TRUE is default.
#' @param reverse Boolean to reverse the color order within the palette.
#' @param ... other parameters that the user wants to pass on to ggplot2::discrete_scale(.)
#' @export
scale_fill_adevinta <- function(palette = "main", discrete = TRUE, reverse = FALSE) {
  pal <- get_palette(palette = palette, reverse = reverse)
  
  if (discrete) {
    discrete_scale("fill", paste0("adevinta", palette), palette = pal)
  } else {
    scale_fill_gradientn(colours = pal(256))
  }
}

#' This function replaces the original ggplot2::scale_x_date(.), so that the date_labels are set to "%b'%y" by default
#' See documentation of ggplot::scale_x_date(.).
#' @export
scale_x_date <- purrr::partial(scale_x_date, date_labels = "%b'%y")


#' labelled numbers for axes.
#' wrapper function for easier text labels definition. Currently, only thousands, millions and percentages supported.
#' See package scales for more options.
#' @param unit string for unit sign to append to numbers. Example: "k", "thousand","M", "millions", "percent", "%"
#' @export
smarter_labels <- function(unit){
  unit <- tolower(unit)
  if(unit == "k" | substr(unit, 1, 4) == "thou"){
    scale <- 1e-3
  } else if(unit == "m" | substr(unit, 1, 3) == "mil"){
    scale <- 1e-6
  } else if(unit %in% c("percent", "percentage", "%")){
    return(scales::percent_format(accuracy = 1))
  } else{
    scales::label_number(suffix = unit, scale = scale, accuracy = 0.1)
  }
}



#' Adevinta ggplot2 theme
#'
#' This theme builds upon ggplot's native theme_minimal to produce a clean layout for data visualisation in Adevinta.
#' @return A function, just like any other theme_*() function in ggplot.
#' @export
#' @examples
#'
#' df <- data.frame(x = rnorm(100), y = rnorm(100), z = sample(c("a", "b"), 100, TRUE))
#'
#' ggplot(df, es(x, y, color = z)) + geom_point() + theme_adevinta()
theme_adevinta <- function(market = "adevinta", text_size_scale = 1, bodytext_size = 18, font_familiy = "Roboto"){
  
  warning("Make sure you install Roboto.ttf before using this functionality. Then use extrafont::font_import(pattern = 'Roboto')")
  
  textcol_adevinta <- "#2900d2"
  textcol_marktplaats <- "#2D3C4D"
  textcol_2dehands <- "#00285A"
  textcol_gumtree <- "#3c3241"
  textcol_kijiji <- "#3E4153"
  
  update_geom_defaults("line", list(size = 0.8))
  update_geom_defaults("point", list(size = 1.5))
  
  bodytext_size <- bodytext_size * text_size_scale
  subtitle_size <- bodytext_size * 1.22 * text_size_scale
  header_size <- bodytext_size * 1.55 * text_size_scale
  
  if(market == "adevinta"){
    textcol <- textcol_adevinta
  } else if(market == "marktplaats"){
    textcol <- textcol_marktplaats
  } else if(market == "gumtree"){
    textcol <- textcol_gumtree
  } else if(market == "2dehands"){
    textcol <- textcol_2dehands
  } else if(market == "kijiji"){
    textcol <- textcol_kijiji
  }
  
  theme_minimal() %+replace%
    theme(
      #grid elements
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      axis.ticks = element_blank(),
      plot.background = element_blank(),
      plot.tag = element_text(
        family = font_familiy,
        size = bodytext_size,
        color = textcol,
        hjust = 0,
        vjust = 1),
      
      plot.title = element_text(
        family = font_familiy,
        size = header_size,
        color = textcol,
        hjust = 0,
        vjust = 1),
      
      plot.subtitle = element_text(
        family = font_familiy,
        size = subtitle_size,
        color = textcol,
        hjust = 0,
      ),
      
      plot.caption = element_text(
        family = font_familiy,
        size = bodytext_size,
        colour = textcol,
        hjust = 0),
      
      axis.title = element_text(
        family = font_familiy,
        size = subtitle_size,
        colour = textcol),
      
      axis.text = element_text(
        family = font_familiy,
        size = bodytext_size,
        colour = textcol),
      
      axis.title.y = element_text(
        vjust = .8,
        hjust = 1, 
        margin = margin(t = 0, r = 20, b = 0, l = 0),
        angle = .45),
      
      axis.title.x = element_text(
        hjust = .5
        ),
      
      legend.position = "right",
      legend.direction = "vertical",
      legend.title = element_text(colour = textcol, size = bodytext_size),
      legend.margin = ggplot2::margin(grid::unit(0,"cm")),
      legend.text = element_text(colour = textcol, size = subtitle_size),
      legend.key.height = grid::unit(0.8, "cm"),
      legend.key.width = grid::unit(0.4, "cm"),
    )
}


#' to render/export ggplot graph
#'
#' @return a visual file in the working directory
#' @export
make_plot <- function(filename, plot = last_plot(), height = 1080, scale = 1.777, unit = 'px') {
  path <- file.path(getwd(), "visualisations")
  dir.create(path, showWarnings = FALSE)
  ggplot2::ggsave(
    plot = plot,
    filename = paste0(filename),
    path = path,
    dpi = "retina",
    height = height,
    width = height*scale,
    units = unit,
    limitsize = TRUE
  )
  message("Plot saved in ~/visualisations")
}

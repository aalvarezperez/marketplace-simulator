#' @title Pizza color wheel
#' 
#' @description
#' This function displays a color wheel with specified colors
#' 
#' @details
#' This function is based on the \code{\link{pie}} function
#' 
#' @param colors a vector with R color names of colors in hexadecimal notation
#' @param bg background color of the plot. Default \code{"gray95"}
#' @param border color of the border separating the pizza slices
#' @param initialise.angle integer value indicating the start angle (in degrees) for
#' the slices
#' @param cex numeric value indicating the character expansion of the labels
#' @param lty argument passed to \code{\link{polygon}} which draws each slice
#' @param labcol color for the labels (i.e. names of the colors)
#' @param \dots graphical parameters (\code{\link{par}}) can be given as
#' argument to \code{pizza}
#' @author Gaston Sanchez
#' @seealso \code{\link{wheel}}
#' @export
#' @examples
#'
#' # pizza color wheel for rainbow colors
#' pizza(rainbow(7))
#' 
#' # pizza color wheel for tomato (18 colors)
#' pizza(setColors("tomato", 18), bg = "gray20", cex = 0.7)
#'
pizza <-
  function(colors, bg = "gray95", border = NA, 
           initialise.angle = 105, cex = 0.8, lty = 1, labcol = NULL, ...)
  {
    n <- length(colors)
    x <- rep(1, n)
    x <- c(0, cumsum(x)/sum(x))
    dx <- diff(x)
    nx <- length(dx)
    # set colors
    labels = colors
    # 
    if (is.null(labcol))
    {
      if (mean(col2rgb(bg)) > 127)
        labcol = rep("black", n)
      if (mean(col2rgb(bg)) <= 127)
        labcol = rep("white", n)
    }
    # prepare plot window
    par(bg = bg)
    plot.new()
    pin <- par("pin")
    xlim <- ylim <- c(-1, 1)
    if (pin[1L] > pin[2L]) 
      xlim <- (pin[1L]/pin[2L]) * xlim
    else ylim <- (pin[2L]/pin[1L]) * ylim
    dev.hold()
    on.exit(dev.flush())
    plot.window(xlim, ylim, "", asp = 1)
    # get ready to plot
    border <- rep(border, length.out = nx)
    if (is.null(border[1]))
      border <- rep(bg, length.out = nx)
    lty <- rep(lty, length.out = nx)
    angle <- rep(45, length.out = nx)
    radius = seq(1, 0, by=-1/n)[1:n]
    twopi <- -2 * pi
    t2xy <- function(t, rad) {
      t2p <- twopi * t + initialise.angle * pi/180
      list(x = rad * cos(t2p), y = rad * sin(t2p))
    }
    # plot colored segments
    for (i in 1L:nx)
    {
      n <- max(2, floor(200 * dx[i]))
      P <- t2xy(seq.int(x[i], x[i + 1], length.out = n), rad=radius[1])
      polygon(c(P$x, 0), c(P$y, 0), angle = angle[i], 
              border = border[i], col = colors[i], lty = lty[i])
      P <- t2xy(mean(x[i + 0:1]), rad=radius[1])
      lab <- labels[i]
      if (!is.na(lab) && nzchar(lab)) {
        adjs = 0.5
        if (P$x > 1e-08) adjs <- 0
        if (P$x < -1e-08) adjs <- 1
        lines(c(1, 1.05) * P$x, c(1, 1.05) * P$y, col=labcol[i])
        text(1.1 * P$x, 1.1 * P$y, labels[i], xpd = TRUE, 
             adj = adjs, cex=cex, col=labcol[i], ...)
      }
    }
    invisible(NULL)
  }

#' @title Tetradic Color Scheme
#' 
#' @description
#' Tetradic color schemes uses four colors arranged into two complementary
#' pairs.
#'
#' @details
#' The tetradic colors are obtained following a color wheel with 12 colors, each
#' one spaced at 30 degrees from each oter.
#' 
#' @param color an R color name or a color in hexadecimal notation
#' @param plot logical value indicating whether to plot a color wheel with the
#' generated scheme
#' @param bg background color of the plot. Used only when \code{plot=TRUE}
#' @param labcol color for the labels (i.e. names of the colors). Used only when
#' \code{plot=TRUE}
#' @param cex numeric value indicating the character expansion of the labels
#' @param title logical value indicating whether to display a title in the plot.
#' Used only when \code{plot=TRUE}
#' @return A character vector with the given color and the tetradic colors in
#' hexadecimal notation
#' @author Gaston Sanchez
#' @seealso \code{\link{complementary}}, \code{\link{splitComp}},
#' \code{\link{adjacent}}, \code{\link{triadic}}, \code{\link{square}}
#' @export
#' @examples
#' # tetradic colors for 'tomato'
#' tetradic("tomato")
#' 
#' # tetradic colors for 'tomato' with bg='gray20'
#' tetradic("tomato", bg = "gray20")
#'
tetradic <-
  function(color, plot=TRUE, bg="white", labcol=NULL, cex=0.8, title=TRUE)
  {	
    tmp_cols = setColors(color, 12)
    tetrad_colors <- tmp_cols[c(1,3,7,9)]
    
    # plot
    if (plot)
    {
      # labels color
      if (is.null(labcol)) 
      {
        lab_col = rep("", 12)
        if (mean(col2rgb(bg)) > 127)
        {
          lab_col[c(1,3,7,9)] <- "black"
          lab_col[c(2,4,5,6,8,10,11,12)] <- col2HSV(bg)
        } else {
          lab_col[c(1,3,7,9)] <- "white"
          lab_col[c(2,4,5,6,8,10,11,12)] <- col2HSV(bg)
        }
      } else {
        lab_col = rep(labcol, 12)
        if (mean(col2rgb(bg)) > 127)
        {
          lab_col[c(1,3,7,9)] <- labcol
          lab_col[c(2,4,5,6,8,10,11,12)] <- col2HSV(bg)
        } else {
          lab_col[c(1,3,7,9)] <- labcol
          lab_col[c(2,4,5,6,8,10,11,12)] <- col2HSV(bg)
        }
      }	
      # hide non-adjacent colors
      tmp_cols[c(2,4,5,6,8,10,11,12)] <- paste(substr(tmp_cols[c(2,4,5,6,8,10,11,12)],1,7), "0D", sep="")
      pizza(tmp_cols, labcol=lab_col, bg=bg, cex=cex)
      # title
      if (title)
        title(paste("Tetradic colors of: ", tmp_cols[1]), 
              col.main=lab_col[1], cex.main=0.8)
    }
    # result
    tetrad_colors
  }


#'@title Set Colors for a color wheel
#'
#'@description
#'This function set a given number of colors to create a color wheel
#'
#'
#'@param color an R color name or a color in hexadecimal notation
#'@param num integer value indicating how many colors to be added to the wheel
#'@return A character vector with the given color and the set of colors to
#'create a wheel color
#'@author Gaston Sanchez
#'@seealso \code{\link{col2HSV}}
#'@export
#'@examples
#'
#' # create a color wheel based on 'tomato'
#' setColors("tomato", 12)
#' 
#' # set 7 colors for '#3D6DCC'
#' setColors("#3D6DCC", 7)
#'
setColors <-
  function(color, num)
  {
    # convert to RGB
    rgb_col = col2rgb(color)
    # convert to HSV
    hsv_col = rgb2hsv(rgb_col)[,1]
    # get degree
    hue = hsv_col[1]
    sat = hsv_col[2]
    val = hsv_col[3]
    cols = seq(hue, hue + 1, by=1/num)
    cols = cols[1:num]
    cols[cols > 1] <- cols[cols > 1] - 1
    # get colors with hsv
    colors = hsv(cols, sat, val)
    # transparency
    if (substr(color, 1, 1) == "#" && nchar(color) == 9)
    {
      alpha = substr(color, 8, 9)
      colors = paste(colors, alpha, sep="")
    }
    colors
  }

#'@title col2HSV: converts a color to HSV in hexadecimal notation
#'
#'@description
#'col2HSV converts an R color (or a set of colors) into an HSV color model, and
#'then returns the color names in hexadeciaml notation
#'
#'@param color an R color name or a color in hexadecimal notation
#'@return A character vector with the color(s) name(s) in hexadecimal notation
#'@author Gaston Sanchez
#'@seealso \code{\link{wheel}}
#'@export
#'@examples
#'
#' # convert 'tomato'
#' col2HSV("tomato")
#'
col2HSV <-
  function(color)
  {
    # convert to RGB
    rgb_col = col2rgb(color)
    # convert to HSV
    hsv_col = rgb2hsv(rgb_col)
    if (length(color) == 1)
    {
      # get degree
      hue = hsv_col[1]
      sat = hsv_col[2]
      val = hsv_col[3]
      # get colors with hsv
      hex_col = hsv(hue, sat, val)
    }
    if (length(color) > 1)
    {
      hex_col = rep("", length(color))
      for (j in 1:length(color))
      {
        hex_col[j] = hsv(hsv_col[1,j], hsv_col[2,j], hsv_col[3,j])
      }
    }
    hex_col
  }
width = getWidth();
height = getHeight();

old = newArray(width * height);

for (y = 0 ; y < height ; y++) {
    for (x = 0 ; x < width ; x++) {
        
        old[y * width + x] = getPixel(x , y);
        
    }
}

for (y = 1 ; y < height - 1 ; y++) {
    for (x = 1 ; x < width - 1 ; x++) {

        structElem = newArray(5);
        structElem[0] = old[(y-1)*width+x];
        structElem[1] = old[(y)*width+x-1];
        structElem[2] = old[(y)*width+x];
        structElem[3] = old[(y)*width+x+1];
        structElem[4] = old[(y+1)*width+x];

        Array.getStatistics(structElem, min, max, mean, std);
        setPixel(x,y,max - min);
        
    }
}
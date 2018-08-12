width = getWidth();
height = getHeight();

histo = newArray(256);
intensityTotal = 0;

for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
        v = getPixel(x, y);
        histo[v]++;
        intensityTotal += v;
    }
}

threshold = intensityTotal / (width * height);
previousThreshold = threshold - 5;

while (round(threshold) != round(previousThreshold)) {
    objectCount = 0;
    backgroundCount = 0;
    totalObjectIntensity = 0;
    totalBackgroundIntensity = 0;
    for (y = 0 ; y < height ; y++) {
        for (x = 0 ; x < width ; x++) {
            pix = getPixel(x , y);
            if (pix <= threshold) {
                objectCount++;
                totalObjectIntensity += pix;
            } else {
                backgroundCount++;
                totalBackgroundIntensity += pix;
            }
        }
    }
    
    avgObjectIntensity = totalObjectIntensity / objectCount;
    avgBackgroundIntensity = totalBackgroundIntensity / backgroundCount;
    previousThreshold = threshold;
    threshold = (avgObjectIntensity + avgBackgroundIntensity) / 2;
}

for (y = 0 ; y < height ; y++) {
    for (x = 0 ; x < width ; x++) {
        if (getPixel(x , y) <= threshold) {
            setPixel(x , y , 0);
        } else {
            setPixel(x , y , 255);
        }
    }
}
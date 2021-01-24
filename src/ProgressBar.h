#ifndef __PROGRESSBAR_HPP__
#define __PROGRESSBAR_HPP__

#include "include_headers.h"


class pBar {
public:
    pBar(double _neededProgress = 100):neededProgress(_neededProgress){}
    
    void update(double newProgress) {
        currentProgress += newProgress;
        amountOfFiller = (int)((currentProgress / neededProgress)*(double)pBarLength);
    }
    void print() {

        
        auto& str = Rcpp::Rcout;
        
        //auto& str = std::cout;
        
        currUpdateVal %= pBarUpdater.length();
        str << "\r" //Bring cursor to start of line
            << firstPartOfpBar; //Print out first part of pBar
        for (int a = 0; a < amountOfFiller; a++) { //Print out current progress
            str << pBarFiller;
        }
        str << pBarUpdater[currUpdateVal];
        for (int b = 0; b < pBarLength - amountOfFiller; b++) { //Print out spaces
            str << " ";
        }
        str << lastPartOfpBar //Print out last part of progress bar
            << " (" << (int)(100*(currentProgress/neededProgress)) << "%)" //This just prints out the percent
            << std::flush;
        currUpdateVal += 1;
    }
    std::string firstPartOfpBar = "[", //Change these at will (that is why they are public)
        lastPartOfpBar = "]",
        pBarFiller = "#",
        pBarUpdater = "/-\\|";
private:
    int amountOfFiller,
        pBarLength = 50, //I would recommend NOT changing this
        currUpdateVal = 0; //Do not change
    double currentProgress = 0, //Do not change
        neededProgress; 
    };
#endif
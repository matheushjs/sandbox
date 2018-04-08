#ifndef CIRCLE_ROUTINES
#define CIRCLE_ROUTINES

#include <QPainter>
#include <QWidget>
#include <cmath>

void drawPixel8(QPainter &painter, int x, int y){
    painter.drawPoint( x, y);
    painter.drawPoint(-x, y);
    painter.drawPoint( x,-y);
    painter.drawPoint(-x,-y);
    painter.drawPoint( y, x);
    painter.drawPoint(-y, x);
    painter.drawPoint( y,-x);
    painter.drawPoint(-y,-x);
}

void drawCircleTraditional(QWidget *wid, int radius){
    QPainter painter(wid);
    painter.translate(wid->width() / 2, wid->height() / 2);

    int limit = radius / std::sqrt(2);
    for(int i = 0; i <= limit; i++){
        drawPixel8(painter, i, std::sqrt(radius*radius - i*i));
    }
}

#endif // CIRCLE_ROUTINES


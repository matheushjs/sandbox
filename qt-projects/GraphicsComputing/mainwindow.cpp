#include <QPainter>
#include "mainwindow.h"
#include "circle_routines.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent)
{
    this->resize(1024, 720);
    this->move(300, 100);
}

void MainWindow::paintEvent(QPaintEvent *event)
{
    drawCircleTraditional(this, 100);
}

import sys
import os
from PyQt5 import QtWidgets
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QVBoxLayout, QMainWindow
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.uic import loadUi
import pandas as pd

# pip install reportlab
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors

class MainWindow(QDialog):
    def __init__(self):
        super().__init__()
        loadUi("portada.ui", self)
        self.CambiarSegundaPagina.clicked.connect(self.SegundaPagina)
        self.segunda = None # Guardar la instancia de MostrarInformacion

    def SegundaPagina(self):
        widget.setCurrentIndex(widget.currentIndex() + 1)

class segundaPagina(QDialog):
    def __init__(self):
        super().__init__()
        loadUi("dialogoprin.ui", self)

        # Llenar los comboboxes con los nombres de las columnas
        self.NombresCultivos.addItems(["Tomate", "Cebolla", "Frijol", "Fresa",])
        # self.NombresCultivos.currentIndexChanged.connect(self.getColumnaFecha)
        self.tercera = None # Guardar la instancia de MostrarInformacion

        self.BotonRegresar.clicked.connect(self.PrimeraPagina)
        self.BotonSiguiente.clicked.connect(self.TerceraPagina)

    def PrimeraPagina(self):
        widget.setCurrentIndex(widget.currentIndex() - 1)

    def TerceraPagina(self):
        widget.setCurrentIndex(widget.currentIndex() + 1)
    
    def getCultivo (self):
        return self.NombresCultivos.currentText()
    
    def getNumeroMesesCultivar (self):
        return self.NumeroMesesCultivar.value()
    
class terceraPagina(QDialog):
    def __init__(self):
        super().__init__()
        loadUi("prediccion.ui", self)
        self.BotonRegresar.clicked.connect(self.SegundaPagina)
        self.BotonSiguiente.clicked.connect(self.UltimaHoja)

    def SegundaPagina(self):
        widget.setCurrentIndex(widget.currentIndex() - 1)

    def UltimaHoja(self):
        widget.setCurrentIndex(widget.currentIndex() + 1)

class ultimaHoja(QDialog):
    def __init__(self):
        super().__init__()
        loadUi("ultimaHoja.ui", self)
        self.BotonRegresar.clicked.connect(self.terceraPagina)
        self.Imprimir.clicked.connect(self.imprimir) # Llamada a la función para generar el PDF

    def terceraPagina(self):
        widget.setCurrentIndex(widget.currentIndex() - 1)

    def imprimir(self):
        # Crear el objeto canvas para el PDF
        nombre_archivo = "curiosity_san_buena.pdf"
        c = canvas.Canvas(nombre_archivo, pagesize=A4)
        width, height = A4
        
        # Definir márgenes y posición inicial
        x_margin = 50
        y_margin = height - 50
        line_height = 14  # Altura entre líneas de texto
        
        # Añadir título con Times-Bold
        c.setFont("Times-Bold", 16)
        c.drawCentredString(width / 2, y_margin, "Curiosity San Buena")
        y_margin -= 30  # Ajustar margen vertical
        
        # Añadir contenido con Times-Roman
        c.setFont("Times-Roman", 12)
        
        # Nombre del cultivo
        c.drawString(x_margin, y_margin, f"● Nombre cultivo: {segunda.getCultivo()}")
        y_margin -= 20
        
        # Tiempo a cultivar
        c.drawString(x_margin, y_margin, f"● Tiempo a cultivar: {segunda.getNumeroMesesCultivar()} meses")
        y_margin -= 20
        
        # Proyección humedad
        c.drawString(x_margin, y_margin, "● Proyección humedad")
        y_margin -= 15
        c.drawString(x_margin + 20, y_margin, "○ Humedad promedio: 77%")
        y_margin -= line_height
        c.drawString(x_margin + 20, y_margin, "○ Humedad máxima: 80%")
        y_margin -= line_height
        c.drawString(x_margin + 20, y_margin, "○ Humedad mínima: 70%")
        y_margin -= 20
        
        # Proyección precipitación
        c.drawString(x_margin, y_margin, "● Proyección precipitación")
        y_margin -= 15
        c.drawString(x_margin + 20, y_margin, "○ Humedad precipitación: 77%")
        y_margin -= line_height
        c.drawString(x_margin + 20, y_margin, "○ Humedad precipitación: 80%")
        y_margin -= line_height
        c.drawString(x_margin + 20, y_margin, "○ Humedad precipitación: 70%")
        y_margin -= 20
        
        # Proyección temperatura
        c.drawString(x_margin, y_margin, "● Proyección temperatura")
        y_margin -= 15
        c.drawString(x_margin + 20, y_margin, "○ Humedad temperatura: 77%")
        y_margin -= line_height
        c.drawString(x_margin + 20, y_margin, "○ Humedad temperatura: 80%")
        y_margin -= line_height
        c.drawString(x_margin + 20, y_margin, "○ Humedad temperatura: 70%")
        y_margin -= 20
        
        # Proyección calidad del aire
        c.drawString(x_margin, y_margin, "● Proyección calidad del aire")
        y_margin -= 15
        c.drawString(x_margin + 20, y_margin, "○ Humedad calidad del aire: 77%")
        y_margin -= line_height
        c.drawString(x_margin + 20, y_margin, "○ Humedad calidad del aire: 80%")
        y_margin -= line_height
        c.drawString(x_margin + 20, y_margin, "○ Humedad calidad del aire: 70%")
        
        # Guardar el PDF
        c.showPage()
        c.save()
        print("Guardado correctamente")

# Configuración de la aplicación
app = QApplication(sys.argv)
mainwindow = MainWindow()
segunda = segundaPagina()
tercera = terceraPagina()
cuarta = ultimaHoja()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.addWidget(segunda)
widget.addWidget(tercera)
widget.addWidget(cuarta)
widget.setFixedWidth(489)
widget.setFixedHeight(758)
widget.show()

try:
    sys.exit(app.exec_())
except:
    print("Saliendo")
from PyQt5.QtWebEngineWidgets import QWebEngineSettings, QWebEngineView
from PyQt5.QtWidgets import QMainWindow, QWidget


class HelpWindow(QMainWindow):
    def __init__(self, pdf_dir):
        super(QMainWindow, self).__init__()
        self.pdf_dir = pdf_dir

        self.setWindowTitle("User Manual")
        self.setGeometry(0, 28, 1000, 750)
        self.centralWidget = QWidget(self)

        self.webView = QWebEngineView()
        self.webView.settings().setAttribute(QWebEngineSettings.PluginsEnabled, True)
        self.webView.settings().setAttribute(QWebEngineSettings.PdfViewerEnabled, True)
        self.setCentralWidget(self.webView)

    def url_changed(self):
        self.setWindowTitle(self.webView.title())

    def go_back(self):
        self.webView.back()

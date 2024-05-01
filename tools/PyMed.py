import sys
import os
import shutil
import subprocess
import platform
import threading
import time

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QGridLayout,
                             QTextEdit, QHBoxLayout, QFrame, QGroupBox, QStackedLayout, QListWidget, QListView,
                             QSizePolicy, QGraphicsView, QGraphicsScene)
from PyQt5.QtGui import QPixmap, QPalette, QBrush, QColor, QIcon, QImageReader
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QFont

# 设置控制台编码为 UTF-8
os.system("chcp 65001")

# 检查C:\monai_python是否存在
if not os.path.exists("C:\\monai_python"):
    # 如果不存在，则从当前目录复制monai_python到C:\monai_python
    print("please waiting......Copying monai_python folder to C:\\...")
    shutil.copytree(".\\monai_python", "C:\\monai_python")
    print("Copying finished.")
else:
    print("Folder C:\\monai_python already exists.")


# 定义一个ClickableLabel，这样可以为QLabel添加点击事件
class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()


class App(QWidget):
    def __init__(self):
        super().__init__()

        # 设置窗口标题和大小
        self.setWindowTitle(
            'PyMed v0.1 医学影像人工智能工具包 一键部署深度学习环境和启动MONAILabel服务端  完全免费！by urozw@hotmail.com（QQ群号 598250489）')
        self.setWindowIcon(QIcon('image/pymed_head.png'))  # 设置窗口图标
        self.setGeometry(900, 400, 800, 600)  # 设置窗口的位置和大小

        # 设置背景颜色
        palette = QPalette()
        sci_fi_color = QColor(247, 247, 247)
        palette.setColor(QPalette.Background, sci_fi_color)
        self.setPalette(palette)

        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)  # 移除主布局的边距，确保左侧导航栏完全填充

        # 导航部分
        self.nav_widget = QWidget()
        self.nav_widget.setFixedWidth(180)  # Set the width of the navigation bar to 80 pixels
        nav_layout = QVBoxLayout()
        self.nav_widget.setLayout(nav_layout)
        self.nav_widget.setStyleSheet("background-color: #e6f3f7;")  # 设置导航部分的背景颜色

        nav_buttons = [("一键运行", "image/one_click_icon.jpg", self.display_one_click),
                       ("示例程序", "image/example_icon.jpg", self.display_example),
                       ("GPU加速", "image/gpu_icon.png", self.display_gpu),
                       ("3DSlicer客户端", "image/3DSlicer.ico", self.display_client)]  # 添加了一个新的按钮

        for text, icon_path, func in nav_buttons:
            btn = QPushButton(QIcon(icon_path), text, self)
            btn.setIconSize(QSize(24, 24))  # 设置图标大小
            btn.clicked.connect(func)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setFixedHeight(40)  # 设置按钮高度
            btn.setFixedWidth(180)  # 设置按钮宽度
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #4682B4;
                    color: white;
                    font-size: 16px;
                    border: none;
                    padding: 5px;
                    text-align: left;
                    padding-left: 10px;
                }
                QPushButton:hover {
                    background-color: #5A8DBD;
                }
                QPushButton:pressed {
                    background-color: #3B6A97;
                }
            """)  # 设置按钮的样式
            nav_layout.addWidget(btn)

        nav_layout.addStretch(1)  # 添加一个弹簧，使按钮在顶部

        main_layout.addWidget(self.nav_widget)

        # 内容部分
        self.stack = QStackedLayout()

        # 一键运行内容
        one_click_layout = QVBoxLayout()

        # 创建QFont对象并设置字体大小
        font = QFont()
        font.setPointSize(11)  # 设置字体大小为11
        description_label = QLabel("""
<html>
<head>
<style>
    body {
        line-height: 1.2;  /* 设置行高 */
        padding: 10px;     /* 设置内边距 */
        word-break: break-word;  /* 根据程序窗口大小自动换行*/
        font-family: 'Microsoft YaHei', Helvetica, Arial, sans-serif;  /* 设置字体 */
        background-color: #f9f9f9; /* 设置背景颜色 */
        color: #333;       /* 设置文本颜色 */
    }
    p {
        margin-bottom: 20px; /* 设置段落间距 */
    }
</style>
</head>
    <body>
        <b>说明：</b><br>&nbsp;&nbsp;1.PyMed是一种医学影像人工智能工具包（Medical Imaging Artificial Intelligence Toolkit），用于简化在本地系统（不支持win7）部署PyTorch深度学习环境和MONAILabel服务端，利用内置或下载的预训练模型权重进行推断，加速医学影像标注。除GPU升级驱动外，PyMed在虚拟环境中进行，不会对系统造成任何影响。若卸载PyMed，直接删除PyMed文件夹即可。PyMed运行期间，<b>“命令行窗口”需要始终处于打开状态！</b><br>&nbsp;&nbsp;2.PyMed所有功能完全免费！MONAI是开源的医学人工智能框架，版权归MONAI所有。PyMed和上尿路预模型归UROZW所有。<br>&nbsp;&nbsp;3.全身分割模型：自动分割104个器官，由MONAI用SegResNet神经网络训练了1204个CT数据，对大的器官分割效果好，对血管分割不理想。<br>&nbsp;&nbsp;4.上尿路分割模型：自动分割肾上腺、肾脏、肾动脉、肾静脉、肾肿瘤等13个器官，对肾血管、肾肿瘤分割效果好，由UROZW用Unet神经网络训练了164个CT数据（动脉期1mm层厚）。<br>&nbsp;&nbsp;5.本版本PyMed安装的是<b>GPU版PyTorch（CUDA12.1）</b>，若<b>GPU驱动程序>=531.14</b>，则默认用GPU，若不满足则用CTU推断，详见“GPU加速”。AI模型自动分割结果仅供参考。<br>&nbsp;&nbsp;6.欢迎提出建议或反馈bug，E-mail：urozw@hotmail.com或QQ群：598250489。
    </body>
</html>
        """)
        description_label.setWordWrap(True)
        description_label.setFont(font)  # 将字体应用于QLabel
        description_label.setStyleSheet("background-color: #f7f7f7;")  # 设置与GUI背景一致的颜色
        one_click_layout.addWidget(description_label)

        bordered_group = self.create_bordered_group()
        one_click_layout.addWidget(bordered_group)
        one_click_widget = QWidget()
        one_click_widget.setLayout(one_click_layout)
        self.stack.addWidget(one_click_widget)

        # 示例程序内容
        description_layout = self.create_description_layout()
        description_widget = QWidget()
        description_widget.setLayout(description_layout)
        self.stack.addWidget(description_widget)

        # GPU加速内容
        gpu_layout = self.create_gpu_layout()  # 修改为调用新的布局方法
        gpu_widget = QWidget()
        gpu_widget.setLayout(gpu_layout)
        self.stack.addWidget(gpu_widget)

        # 客户端内容
        client_layout = self.create_client_layout()
        client_widget = QWidget()
        client_widget.setLayout(client_layout)
        self.stack.addWidget(client_widget)

        main_layout.addLayout(self.stack)
        self.setLayout(main_layout)

    def display_one_click(self):
        self.stack.setCurrentIndex(0)

    def display_example(self):
        self.stack.setCurrentIndex(1)

    def display_gpu(self):
        self.stack.setCurrentIndex(2)

    def display_client(self):
        self.stack.setCurrentIndex(3)

    def create_description_layout(self):
        layout = QVBoxLayout()

        description_text = QTextEdit(self)
        font = QFont()
        font.setPointSize(11)
        description_text.setFont(font)
        description_text.setText("""
<html>
<head>
<style>
    body {
        line-height: 1.2;  /* 设置行高 */
        padding: 10px;     /* 设置内边距 */
        font-family: 'Microsoft YaHei', Helvetica, Arial, sans-serif;  /* 设置字体 */
        background-color: #f9f9f9; /* 设置背景颜色 */
        color: #333;       /* 设置文本颜色 */
    }
    p {
        margin-bottom: 20px; /* 设置段落间距 */
    }
</style>
</head>
    <body>
&nbsp;&nbsp;运行MOANILabel示例程序，自动下载MONAILabel预训练模型，请单击下方“启动MONAILabel服务端”，复制粘贴下面的命令至“命令行窗口”，按"Enter"键执行：<br>【<b>运行Radiology程序</b>】<br><font color="blue">monailabel start_server --app radiology --studies datasets --conf models segmentation</font><br>&nbsp;&nbsp;其中segmentation可修改为：Deepedit、Deepgrow、Spleen Segmentation、Multi-Stage Vertebra Segmentation等。<br>【<b>运行MONAIBundle程序</b>】<br><font color="blue">monailabel start_server --app monaibundle --studies datasets --conf models spleen_ct_segmentation</font> <br>&nbsp;&nbsp;其中spleen_ct_segmentation可修改为：swin_unetr_btcv_segmentation、prostate_mri_anatomy、wholeBrainSeg_UNEST_segmentation、lung_nodule_ct_detection、wholeBody_ct_segmentation等。<br>&nbsp;&nbsp;<b>备注</b>：MONAIBundle每次启动时，使用MONAI Bundle API从github的Model-Zoo检索模型最新型号的信息。因为github对速率有一定限制，因此MONAIBundle启动较慢，有时会启动失败，这种情况可以多试几次，更好的办法是使用“--conf auth_token”，auth_token是您的github个人访问令牌。如：monailabel start_server --app monaibundle --studies datasets --conf models spleen_ct_segmentation --conf auth_token。个人访问令牌的更多信息请参阅：https://docs.github.com/en/rest/overview/resources-in-the-rest-api#rate-limiting <br> 【<b>同时运行Radiology和Bundle程序</b>】 <br><font color="blue">monailabel start_server --app radiology --studies datasets --conf models segmentation --conf bundles spleen_ct_segmentation</font> <br>【<b>更新MONAILabel</b>】 <br><font color="blue">pip install --upgrade monailabel -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com</font><br>&nbsp;&nbsp;需要更新，或者运行出错时，请运行上面的命令更新MONAILabel。 <br><br>&nbsp;&nbsp;更多信息请参阅：https://monai.io、https://docs.monai.io/projects/label/en/latest/quickstart.html、https://github.com/project-monai/monailabel <br>
    </body>
</html>
        """)

        description_text.setReadOnly(True)
        description_text.setStyleSheet("background-color: #f7f7f7;")
        layout.addWidget(description_text)

        # 修改为以下代码
        cmd_button = QPushButton("启动MONAILabel服务端", self)
        cmd_button.clicked.connect(self.open_cmd)
        cmd_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # 设置按钮的尺寸策略为扩展

        font = QFont()
        font.setPointSize(12)
        cmd_button.setFont(font)
        cmd_button.setFixedWidth(220)
        cmd_button.setFixedHeight(40)

        cmd_button.setStyleSheet("""
            QPushButton {
                background-color: #4682B4;
                color: white;
                font-size: 16px;
                border: none;
                padding: 5px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #5A8DBD;
            }
            QPushButton:pressed {
                background-color: #3B6A97;
            }
        """)

        layout.addWidget(cmd_button)

        return layout

    def create_gpu_layout(self):
        layout = QVBoxLayout()

        description_text = QTextEdit(self)
        font = QFont()
        font.setPointSize(11)
        description_text.setFont(font)
        description_text.setText("""
<html>
<head>
<style>
    body {
        line-height: 1.2;  /* 设置行高 */
        padding: 10px;     /* 设置内边距 */
        font-family: 'Microsoft YaHei', Helvetica, Arial, sans-serif;  /* 设置字体 */
        background-color: #f9f9f9; /* 设置背景颜色 */
        color: #333;       /* 设置文本颜色 */
    }
    p {
        margin-bottom: 20px; /* 设置段落间距 */
    }
</style>
</head>
    <body>
&nbsp;&nbsp;&nbsp;&nbsp;MONAI建立在PyTorch之上，PyTorch分为CPU版和GPU版。本版本PyMed安装的是<b>GPU版PyTorch（CUDA12.1）</b>，若<b>GPU驱动程序>=531.14</b>，则默认用GPU，若不满足则用CTU。<br>&nbsp;&nbsp;&nbsp;&nbsp;若要加快模型推断，或者训练模型（包括微调模型或从头开始训练），电脑需要配有NVIDIA GPU。<br>&nbsp;&nbsp;&nbsp;&nbsp;下面的步骤，除升级驱动程序外，其余步骤均是在PyMed虚拟环境下进行，不会对系统造成任何影响。<br>【<b>当前配置</b>】<br>&nbsp;&nbsp;&nbsp;&nbsp;查看本电脑的配置和PyMed目前可以识别的设备。若提示有NVIDIA GPU，可以进行后面的NVIDIA驱动升级和更新PyTorch。<br>【<b>升级驱动</b>】<br>&nbsp;&nbsp;&nbsp;&nbsp;升级NVIDIA驱动程序，NVIDIA驱动程序版本决定CUDA版本。此步骤不是必须，但升级后可以安装最新的CUDA版本（https://www.nvidia.cn/Download/index.aspx?lang=cn）。在安装最新的驱动版本前，无需手动卸载旧版。<br>【<b>更新PyTorch</b>】<br>&nbsp;&nbsp;&nbsp;&nbsp;首先卸载PyMed的CPU版PyTorch，然后依据当前的NVIDIA驱动版本，安装匹配的GPU版PyTorch。当使用pip从PyTorch的官方网站安装PyTorch时，它会自动包含与之匹配的CUDA runtime和cuDNN。所以，不需要单独下载和安装CUDA和cuDNN。若下载PyTorch出错，通常是网络不稳定所致，多试几次。<br>&nbsp;&nbsp;&nbsp;&nbsp;CUDA（统一计算设备架构）是由NVIDIA公司开发的一种并行计算平台和编程模型。cuDNN（CUDA深度神经网络库）是由NVIDIA开发的一个针对深度神经网络的GPU加速库。cuDNN提供了高效的运行时支持，包括前向和反向卷积、池化、归一化和张量转换等。深度学习PyTorch环境的部署，主要是让GPU版的PyTorch应用CUDA和cuDNN，加速计算。<br>【<b>GPU配置</b>】<br>&nbsp;&nbsp;&nbsp;&nbsp;查看PyMed当前深度学习环境的配置信息，包括GPU型号、CUDA型号、python和PyTorch版本等。<br>【<b>启动cmd</b>】<br>&nbsp;&nbsp;&nbsp;&nbsp;打开深度学习环境的命令行窗口。<br>【<b>启动Jupyter Lab</b>】<br>&nbsp;&nbsp;&nbsp;&nbsp;打开深度学习环境的Jupyter Lab，可以安装和运行其他的深度学习项目。<br><br>&nbsp;&nbsp;&nbsp;&nbsp;<b>备注</b>：配置成功后，MONAILabel会用GPU推断和训练，不需额外设置（因为device = torch.device("cuda" if torch.cuda.is_available() else "cpu")）。另外，安装过程可能会因系统配置、显卡型号和其他因素而不同，具体请参考NVIDIA官方文档。<br>
    </body>
</html>
        """)

        description_text.setReadOnly(True)
        description_text.setStyleSheet("background-color: #f7f7f7;")
        layout.addWidget(description_text)

        # 创建按钮的水平布局
        button_layout_top = QHBoxLayout()
        button_layout_bottom = QHBoxLayout()

        # 定义按钮信息
        button_info_top = [
            ("当前配置", "gpu\\gpu_check.py"),
            ("升级驱动", "gpu\\gpu_update.py"),
            ("更新PyTorch", "gpu\\gpu_torch.py"),
            ("GPU配置", "gpu\\gpu_config.py")
        ]
        button_info_bottom = [
            ("启动命令行窗口", "open_cmd2.py"),
            ("启动Jupyter Lab", "jupyter_lab.py")
        ]

        # 创建上排按钮
        for text, script in button_info_top:
            btn = self.create_button(text, script)
            button_layout_top.addWidget(btn)

        # 创建下排按钮，并添加空白占位符
        button_layout_bottom.addStretch(2)  # 添加空白区域
        for text, script in button_info_bottom:
            btn = self.create_button(text, script)
            button_layout_bottom.addWidget(btn)

        # 将按钮布局添加到主布局
        layout.addLayout(button_layout_top)
        layout.addLayout(button_layout_bottom)

        return layout

    def create_button(self, text, script):
        btn = QPushButton(text, self)
        btn.clicked.connect(lambda checked, s=script: self.run_script(s))
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn.setFixedHeight(40)
        btn.setFixedWidth(180)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #4682B4;
                color: white;
                font-size: 16px;
                border: none;
                padding: 5px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #5A8DBD;
            }
            QPushButton:pressed {
                background-color: #3B6A97;
            }
        """)
        return btn

    def create_client_layout(self):
        layout = QGridLayout()
        layout.setSpacing(20)  # 设置图片之间的间隔为20

        images = ["3DSlicer_01.jpg", "3DSlicer_02.jpg", "3DSlicer_03.jpg", "3DSlicer_04.jpg"]
        titles = ["下载3DSlicer", "下载MONAILabel插件", "连接服务端并拖入CT数据，运行自动分割", "分割结果（点击图片放大）"]

        for i, (image_name, title) in enumerate(zip(images, titles)):
            label = ClickableLabel(self)
            pixmap = QPixmap(os.path.join("image", image_name)).scaled(350, 400, Qt.KeepAspectRatio)
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)

            def handler(img=image_name):
                return lambda: self.open_image(img)

            label.clicked.connect(handler())

            # 创建一个框架并将标签添加到框架中
            frame = QGroupBox(title, self)
            frame_layout = QVBoxLayout()
            frame_layout.addWidget(label)
            frame.setLayout(frame_layout)

            row = i // 2  # 计算行号：0或1
            col = i % 2  # 计算列号：0或1
            layout.addWidget(frame, row, col)

        wrapper = QWidget()  # 创建一个包装器部件来容纳布局
        wrapper.setLayout(layout)
        final_layout = QVBoxLayout()  # 创建一个垂直布局来确保我们的网格布局居中
        final_layout.addWidget(wrapper, 0, Qt.AlignCenter)
        return final_layout

    def open_image(self, image_name):
        subprocess.Popen(["explorer", os.path.join("image", image_name)])

    def create_bordered_group(self):
        bordered_group = QGroupBox("一键启动MONAILabel服务端")
        grid = QGridLayout()

        scripts = [("main_full.py", "启动全身分割模型", "image/image1.png"),
                   ("main_uut.py", "启动上尿路分割模型", "image/image2.jpg"),
                   ("main_full_uut.py", "启动全身和上尿路分割模型", "image/image3.jpg")]

        for i, (script, btn_text, image_path) in enumerate(scripts):
            pixmap = QPixmap(image_path).scaled(300, 200, Qt.KeepAspectRatio)
            # label = QLabel(self)
            label = ClickableLabel(self)
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)

            # 点击图片打开
            image_name = os.path.basename(image_path)

            def handler(img=image_name):
                return lambda: self.open_image(img)

            label.clicked.connect(handler())

            grid.addWidget(label, 0, i)

            btn = QPushButton(btn_text, self)
            btn.clicked.connect(lambda checked, s=script: self.run_script(s))
            grid.addWidget(btn, 1, i)

            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # 设置按钮的尺寸策略为扩展
            font = QFont()
            font.setPointSize(12)
            btn.setFont(font)
            btn.setFixedHeight(40)

            # 设置按钮的样式，确保与之前的代码相同
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #4682B4;
                    color: white;
                    font-size: 16px;
                    border: none;
                    padding: 5px;
                    margin: 5px;
                }
                QPushButton:hover {
                    background-color: #5A8DBD;
                }
                QPushButton:pressed {
                    background-color: #3B6A97;
                }
            """)

        bordered_group.setLayout(grid)
        return bordered_group

    def run_script(self, script_name):
        process = subprocess.Popen(["Scripts\\python.exe", script_name])

    def open_cmd(self):
        subprocess.Popen(["Scripts\\python.exe", "open_cmd.py"])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())

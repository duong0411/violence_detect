import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# Thông tin tài khoản Gmail của bạn
sender_email = "duong04112000@gmail.com"
sender_password = "pqfo bpvy zvfm abpk"

# Địa chỉ email người nhận
receiver_email = "thankinhchochet11986@gmail.com"

# Tạo một email
msg = MIMEMultipart()
msg["From"] = sender_email
msg["To"] = receiver_email
msg["Subject"] = "Subject of the Email"

# Thêm nội dung email (nếu cần)
message = "This is the email message body."
msg.attach(MIMEText(message, "plain"))

# Đính kèm ảnh
image_path = "C:/Users/Admin/PycharmProjects/pythonProject5/IMG_0798.JPG"
with open(image_path, "rb") as img_file:
    image = MIMEImage(img_file.read())
    msg.attach(image)

# Kết nối đến máy chủ SMTP của Gmail
server = smtplib.SMTP("smtp.gmail.com", 587)
server.starttls()

# Đăng nhập vào tài khoản Gmail của bạn
server.login(sender_email, sender_password)

# Gửi email
server.sendmail(sender_email, receiver_email, msg.as_string())

# Đăng xuất khỏi tài khoản Gmail và đóng kết nối
server.quit()

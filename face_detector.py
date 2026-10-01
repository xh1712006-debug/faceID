import cv2

def main():
    print("Dang khoi dong camera...")
    # Tải mô hình nhận diện khuôn mặt có sẵn của OpenCV
    # OpenCV đã cung cấp sẵn file haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Sử dụng cv2.CAP_DSHOW trên Windows để tránh các lỗi không tương thích hoặc kẹt camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Khong the mo camera. Vui long kiem tra lai thiet bi!")
        return

    print("Camera da mo. Nhan phim 'q' tren cua so video de thoat.")

    while True:
        # Đọc từng khung hình từ camera
        ret, frame = cap.read()
        if not ret:
            print("Loi: Khong the nhan du lieu tu camera")
            break

        # Chuyển đổi khung hình sang ảnh xám (gray scale) để nhận diện khuôn mặt dễ hơn và nhanh hơn
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Nhận diện khuôn mặt trong khung hình
        # scaleFactor và minNeighbors là các tham số dùng để tinh chỉnh độ nhạy
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )

        # Đếm số người (chính là số khuôn mặt phát hiện được)
        num_faces = len(faces)

        # Vẽ hình chữ nhật xung quanh mỗi khuôn mặt tìm thấy
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Hiển thị số lượng người trên màn hình
        cv2.putText(
            frame, 
            f'So nguoi: {num_faces}', 
            (10, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 0, 255), # Màu đỏ (B, G, R)
            2
        )

        # Hiển thị video
        cv2.imshow('Nhan dien khuon mat', frame)

        # Chờ 1 mili giây, nếu người dùng bấm phím 'q' thì thoát vòng lặp
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Dọn dẹp: Giải phóng camera và đóng tất cả cửa sổ
    cap.release()
    cv2.destroyAllWindows()
    print("Da dong chuong trinh.")

if __name__ == "__main__":
    main()

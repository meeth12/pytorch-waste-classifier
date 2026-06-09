//ContentView

import SwiftUI



struct ContentView: View {
    @State private var selectedImage: UIImage? = nil
    @State private var showPhotoLibrary = false
    @State private var showCamera = false
    
    @State private var predictedCategory: String = "No result yet"
    @State private var confidenceText: String = "-"
    @State private var userMessage: String = "Select or take a photo to begin."
    @State private var errorMessage: String? = nil
    
    @State private var isLoading = false

    //Server address
    private let serverURLString = "http://192.168.0.216:8000/classify"
    
    private func classifySelectedImage() {
        
        errorMessage = nil
        
        let startTime = Date()
        print("⏱️ Classify started at \(startTime)")
        
        guard let selectedImage = selectedImage else {
            errorMessage = "Please select an image before classifying."
            return
        }
        
        guard let url = URL(string: serverURLString) else {
            errorMessage = "Invalid server URL."
            return
        }
        
        // Convert image to JPEG for upload
        guard let imageData = selectedImage.jpegData(compressionQuality: 0.8) else {
            errorMessage = "Could not convert image for upload."
            return
        }
        
        isLoading = true
        userMessage = "Sending image to server..."
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        // Multipart/form-data body
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        let httpBody = createMultipartBody(
            boundary: boundary,
            imageData: imageData,
            fieldName: "image",
            fileName: "upload.jpg",
            mimeType: "image/jpeg"
        )
        
        URLSession.shared.uploadTask(with: request, from: httpBody) { data, response, error in
            
            let elapsed = Date().timeIntervalSince(startTime)
            print(String(format: "📡 UploadTask finished in %.3f seconds", elapsed))
            
            DispatchQueue.main.async {
                isLoading = false
            }
            
            if let error = error {
                DispatchQueue.main.async {
                    errorMessage = "Network error: \(error.localizedDescription)"
                    userMessage = "Could not contact the server."
                }
                return
            }
            
            guard let httpResponse = response as? HTTPURLResponse else {
                DispatchQueue.main.async {
                    errorMessage = "Invalid server response."
                    userMessage = "No valid response was received."
                }
                return
            }
            
            guard let data = data else {
                DispatchQueue.main.async {
                    errorMessage = "No data received from server."
                    userMessage = "The server returned an empty response."
                }
                return
            }
            
            
            if !(200...299).contains(httpResponse.statusCode) {
                let serverMessage = parseServerErrorMessage(from: data) ?? "Server returned status \(httpResponse.statusCode)."
                DispatchQueue.main.async {
                    errorMessage = serverMessage
                    userMessage = "Classification failed."
                }
                return
            }
            
            DispatchQueue.main.async {
                do {
                    let decoded = try JSONDecoder().decode(ClassificationResponse.self, from: data)

                    predictedCategory = decoded.predictedClass ?? "Unknown"

                    if let confidence = decoded.confidence {
                        confidenceText = String(format: "%.2f%%", confidence * 100)
                    } else {
                        confidenceText = "-"
                    }

                    if let message = decoded.message, !message.isEmpty {
                        userMessage = message
                    } else {
                        userMessage = "Classification completed successfully."
                    }
                } catch {
                    errorMessage = "Could not read server response."
                    userMessage = "The app received data, but it was not in the expected format."
                }

                isLoading = false
            }
        }.resume()
    }
    
    private func parseServerErrorMessage(from data: Data) -> String? {
        // Try a simple JSON error message first
        if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let message = json["error"] as? String {
                return message
            }
            if let message = json["message"] as? String {
                return message
            }
        }
        return nil
    }
    
    private func createMultipartBody(
        boundary: String,
        imageData: Data,
        fieldName: String,
        fileName: String,
        mimeType: String
    ) -> Data {
        var body = Data()
        let lineBreak = "\r\n"
        
        body.append("--\(boundary)\(lineBreak)".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"\(fieldName)\"; filename=\"\(fileName)\"\(lineBreak)".data(using: .utf8)!)
        body.append("Content-Type: \(mimeType)\(lineBreak)\(lineBreak)".data(using: .utf8)!)
        body.append(imageData)
        body.append(lineBreak.data(using: .utf8)!)
        body.append("--\(boundary)--\(lineBreak)".data(using: .utf8)!)
        
        return body
    }
    
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 16) {
                    
                    Text("Waste Classifier")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text("Select or take a photo, then classify it.")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    // Image preview area
                    Group {
                        if let image = selectedImage {
                            Image(uiImage: image)
                                .resizable()
                                .scaledToFit()
                                .frame(maxWidth: .infinity)
                                .frame(height: 250)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12)
                                        .stroke(Color.gray.opacity(0.3), lineWidth: 1)
                                )
                        } else {
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color.gray.opacity(0.1))
                                .frame(height: 250)
                                .overlay(
                                    VStack(spacing: 8) {
                                        Image(systemName: "photo")
                                            .font(.system(size: 40))
                                            .foregroundColor(.gray)
                                        Text("No image selected")
                                            .foregroundColor(.gray)
                                    }
                                )
                        }
                    }
                    
                    // Buttons
                    VStack(spacing: 12) {
                        Button(action: {
                            errorMessage = nil
                            showPhotoLibrary = true
                        }) {
                            Label("Choose Image", systemImage: "photo.on.rectangle")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        
                        Button(action: {
                            errorMessage = nil
                            if UIImagePickerController.isSourceTypeAvailable(.camera) {
                                showCamera = true
                            } else {
                                errorMessage = "Camera is not available on this device/simulator."
                            }
                        }) {
                            Label("Take Photo", systemImage: "camera")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        
                        Button(action: {
                            classifySelectedImage()
                        }) {
                            Label("Classify", systemImage: "checkmark.circle")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(selectedImage == nil || isLoading)
                    }
                    
                    // Result area (placeholder for now)
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Result")
                            .font(.headline)
                        
                        HStack {
                            Text("Category:")
                                .fontWeight(.semibold)
                            Spacer()
                            Text(predictedCategory)
                        }
                        
                        HStack {
                            Text("Confidence:")
                                .fontWeight(.semibold)
                            Spacer()
                            Text(confidenceText)
                        }
                        
                        Text(userMessage)
                            .foregroundColor(.secondary)
                            .padding(.top, 4)
                    }
                    .padding()
                    .background(Color.gray.opacity(0.08))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    
                    // Error area
                    if let errorMessage = errorMessage {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Error")
                                .font(.headline)
                            Text(errorMessage)
                                .foregroundColor(.red)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .background(Color.red.opacity(0.08))
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                    
                    Spacer(minLength: 20)
                }
                .padding()
            }
        }
        .sheet(isPresented: $showPhotoLibrary) {
            ImagePicker(
                sourceType: .photoLibrary,
                selectedImage: $selectedImage
            )
        }
        .sheet(isPresented: $showCamera) {
            ImagePicker(
                sourceType: .camera,
                selectedImage: $selectedImage
            )
        }
    }
}

#Preview {
    ContentView()
}

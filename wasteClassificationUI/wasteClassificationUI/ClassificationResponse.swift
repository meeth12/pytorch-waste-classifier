//ClassificationResponse

import Foundation

struct ClassificationResponse: Decodable, Sendable {
    let predictedClass: String?
    let confidence: Double?
    let message: String?
}

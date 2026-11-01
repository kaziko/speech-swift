import MLX
import XCTest

@testable import OmniVoiceTTS

final class OmniVoiceReferenceTests: XCTestCase {
    func testTokenCountReadsTheTimeAxis() {
        let reference = OmniVoiceReference(tokens: MLXArray.zeros([1, 8, 50], type: Int32.self))
        XCTAssertEqual(reference.tokenCount, 50)
    }
}

// Uso: swift vision.swift <imagem> [mascara-saida.png]
// Imprime JSON com olhos/boca em pixels (origem no topo) e, se pedido, salva a máscara de pessoa.
import Foundation
import Vision
import CoreImage
import ImageIO
import UniformTypeIdentifiers

let args = CommandLine.arguments
let url = URL(fileURLWithPath: args[1])
let src = CGImageSourceCreateWithURL(url as CFURL, nil)!
let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)!
let W = Double(cg.width), H = Double(cg.height)
let handler = VNImageRequestHandler(cgImage: cg)

let face = VNDetectFaceLandmarksRequest()
try handler.perform([face])
var out: [String: Any] = ["width": W, "height": H]
if let f = face.results?.first, let lm = f.landmarks {
  func center(_ r: VNFaceLandmarkRegion2D?) -> [Double]? {
    guard let p = r?.pointsInImage(imageSize: CGSize(width: W, height: H)), !p.isEmpty else { return nil }
    let x = p.map { Double($0.x) }.reduce(0, +) / Double(p.count)
    let y = p.map { Double($0.y) }.reduce(0, +) / Double(p.count)
    return [x, H - y]
  }
  out["leftEye"] = center(lm.leftPupil ?? lm.leftEye)
  out["rightEye"] = center(lm.rightPupil ?? lm.rightEye)
  out["mouth"] = center(lm.outerLips)
  let b = f.boundingBox
  out["faceBox"] = [b.minX * W, (1 - b.maxY) * H, b.width * W, b.height * H]
}
if args.count > 2 {
  let seg = VNGeneratePersonSegmentationRequest()
  seg.qualityLevel = .accurate
  seg.outputPixelFormat = kCVPixelFormatType_OneComponent8
  try handler.perform([seg])
  if let buf = seg.results?.first?.pixelBuffer {
    var ci = CIImage(cvPixelBuffer: buf)
    ci = ci.transformed(by: CGAffineTransform(scaleX: W / ci.extent.width, y: H / ci.extent.height))
    let ctx = CIContext()
    let outURL = URL(fileURLWithPath: args[2])
    try ctx.writePNGRepresentation(of: ci, to: outURL, format: .L8, colorSpace: CGColorSpaceCreateDeviceGray())
  }
}
let data = try JSONSerialization.data(withJSONObject: out, options: [.sortedKeys])
print(String(data: data, encoding: .utf8)!)

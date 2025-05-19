import { v2 as cloudinary } from "cloudinary";

cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
});

// Function to generate a data URL from a buffer
function bufferToDataUrl(buffer: Buffer): string {
  // Convert buffer to base64
  const base64 = buffer.toString('base64');
  return `data:image/jpeg;base64,${base64}`;
}

export async function uploadImage(
  file: Buffer
): Promise<{ url: string; publicId: string }> {
  try {
    return await new Promise((resolve, reject) => {
      const uploadOptions = {
        folder: "cancer-scans",
        resource_type: "image" as "image",
      };

      cloudinary.uploader
        .upload_stream(uploadOptions, (error, result) => {
          if (error || !result) {
            return reject(error || new Error("Upload failed"));
          }
          resolve({
            url: result.secure_url,
            publicId: result.public_id,
          });
        })
        .end(file);
    });
  } catch (error) {
    console.error("Cloudinary upload failed:", error);
    console.log("Using local data URL as fallback...");
    
    // If Cloudinary upload fails, use a data URL as fallback
    // This allows development to continue without Cloudinary credentials
    const dataUrl = bufferToDataUrl(file);
    const mockPublicId = `local-${Date.now()}`;
    
    return {
      url: dataUrl,
      publicId: mockPublicId,
    };
  }
}

export async function deleteImage(publicId: string): Promise<void> {
  await cloudinary.uploader.destroy(publicId);
}

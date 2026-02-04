import React from "react";

export default function AboutPage() {
  return (
    <div className="container mx-auto py-10">
      <h1 className="text-3xl font-bold mb-4">About the Project</h1>
      <p className="mb-4">
        This project aims to develop an advanced AI-based system for the early
        detection of lung cancer using machine learning algorithms applied to
        medical imaging data.
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">Objectives</h2>
      <p className="mb-4">
        The objective is to design and implement a system that can accurately
        classify cancerous nodules in CT scans, assisting healthcare
        professionals in early diagnosis. By leveraging state-of-the-art machine
        learning techniques and medical imaging technologies, the project seeks
        to enhance diagnostic capabilities within the healthcare sector.
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">Context</h2>
      <p className="mb-4">
        This research integrates artificial intelligence, machine learning, and
        medical diagnostics, bridging the gap between engineering technology and
        healthcare applications. The project focuses on improving early cancer
        detection through precision and reliability.
      </p>

      <h2 className="text-2xl font-semibold mt-6 mb-2">
        Limitations & Recommendations
      </h2>
      <div className="p-4 border rounded bg-yellow-50">
        <h3 className="font-bold">IMPORTANT DISCLAIMER</h3>
        <p className="mt-2 text-sm">
          Beta Version - Accuracy Under Development
        </p>
        <ul className="list-disc ml-5 mt-2 text-sm">
          <li>
            LIMITATIONS: This AI tool is in active development and testing.
            Prediction accuracy may vary and is not guaranteed. Results should
            NOT replace professional medical consultation. Not approved for
            clinical diagnostic use.
          </li>
          <li>
            RECOMMENDATIONS: Use only as a preliminary screening tool. Consult
            licensed healthcare providers for diagnosis. Seek immediate medical
            attention for urgent concerns.
          </li>
        </ul>
        <p className="mt-2 text-sm">
          Acknowledgment: By using this service, you acknowledge these
          limitations.
        </p>
      </div>
    </div>
  );
}

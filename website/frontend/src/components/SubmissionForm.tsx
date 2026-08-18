import React, { useState } from 'react';
import { submitModel } from '../api/client';

export const SubmissionForm: React.FC = () => {
  const [form, setForm] = useState({
    model_id: '',
    architecture: '',
    parameters: '',
    contact_email: '',
    latency_p50_ms: '',
    throughput: '',
  });
  const [message, setMessage] = useState('');

  const submit = async () => {
    try {
      const result = await submitModel({
        model_id: form.model_id,
        architecture: form.architecture || undefined,
        parameters: form.parameters ? Number(form.parameters) : undefined,
        contact_email: form.contact_email || undefined,
        metrics: {
          ...(form.latency_p50_ms ? { latency_p50_ms: Number(form.latency_p50_ms) } : {}),
          ...(form.throughput ? { throughput: Number(form.throughput) } : {}),
        },
      });
      setMessage(
        `Submission ${result.submission.submission_id} received — pending review.`
      );
    } catch (err) {
      console.error(err);
      setMessage('Submission failed. Please check the fields.');
    }
  };

  const field = (key: keyof typeof form, label: string, placeholder = '') => (
    <label className="block mb-2">
      <span className="text-sm font-medium">{label}</span>
      <input
        value={form[key]}
        onChange={(e) => setForm({ ...form, [key]: e.target.value })}
        placeholder={placeholder}
        className="mt-1 w-full border rounded px-3 py-2"
      />
    </label>
  );

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-4">Submit a model result</h3>
      <div className="grid grid-cols-2 gap-3">
        {field('model_id', 'Model id *', 'e.g. org/model-name')}
        {field('architecture', 'Architecture', 'cnn / transformer')}
        {field('parameters', 'Parameters', 'e.g. 7000000000')}
        {field('contact_email', 'Contact email')}
        {field('latency_p50_ms', 'Latency P50 (ms)')}
        {field('throughput', 'Throughput (samples/s)')}
      </div>
      <button
        onClick={submit}
        className="mt-4 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
      >
        Submit
      </button>
      {message && <p className="mt-3 text-sm text-gray-600">{message}</p>}
    </div>
  );
};

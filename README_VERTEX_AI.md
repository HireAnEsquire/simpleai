# Setting Up Google Cloud Vertex AI

If you need enterprise features, compliance guarantees, or just prefer to use Google Cloud Platform's Vertex AI instead of Google AI Studio for Gemini, follow these steps to configure your project and authenticate the `simpleai` package.

## Step 1: Create a Google Cloud Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Click the project drop-down menu at the top of the page.
3. Click **New Project** in the upper right of the modal.
4. Enter a **Project name** (e.g., `simpleai-vertex`).
5. Note your **Project ID** (e.g., `simpleai-vertex-123456`) - you will need this for the `simpleai` configuration.
6. Click **Create**.
7. Once created, make sure the project is selected in the top drop-down menu.

## Step 2: Enable the Vertex AI API

1. In the Cloud Console, use the top search bar to search for "Vertex AI API" or navigate to **APIs & Services > Library**.
2. Click on the **Vertex AI API** from the results.
3. Click the **Enable** button.
4. Wait a moment for the API to be enabled for your project. (You may also need to set up a billing account if you haven't already).

## Step 3: Create a Service Account

To access Vertex AI securely, you need to create a Service Account that your application will use.

1. Navigate to **IAM & Admin > Service Accounts** in the left sidebar menu.
2. Click **+ Create Service Account** at the top.
3. Provide a **Service account name** (e.g., `simpleai-bot`).
4. (Optional) Provide a description, then click **Create and Continue**.
5. In the **Grant this service account access to project** section, click the **Select a role** dropdown.
6. Search for and select the **Vertex AI User** role. (This gives the account the permissions needed to use models).
7. If you want true binary file uploads through `simpleai` on Vertex AI, also grant a storage role on your upload bucket:
   - **Storage Object User** for read/write/delete object access, or
   - a custom role with at least `storage.objects.create`, `storage.objects.get`, and `storage.objects.delete`.
7. Click **Continue**, then **Done**.

## Step 4: Generate and Download the Credentials File

1. Back on the **Service Accounts** page, click the email address of the service account you just created.
2. Go to the **Keys** tab.
3. Click **Add Key > Create new key**.
4. Keep **JSON** selected as the key type and click **Create**.
5. A JSON file will automatically download to your computer.
6. **Important:** Move this JSON file to a secure location on your machine or server. **Do not commit this file to source control (e.g., do not push it to GitHub)!** 

*Example secure location: `/Users/yourname/.gcp/simpleai-vertex-key.json`*

## Step 5: Configure Application Default Credentials (ADC)

The Google GenAI SDK automatically detects authentication via an environment variable called `GOOGLE_APPLICATION_CREDENTIALS`.

You need to set this environment variable to point to the absolute path of the JSON key file you just downloaded.

### For local development (.env / terminal):
If you're using `.env` or running scripts manually, set the variable in your terminal or `.env` file:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/simpleai-vertex-key.json"
```

### For Docker/Servers:
Ensure the file is copied or mounted to the container/server and the environment variable is set in the runtime environment.

## Step 6: Configure SimpleAI

Finally, configure `simpleai` to use Vertex AI instead of the standard API. 

You need to provide your **Project ID** from Step 1 and the **Location/Region** you want to use (e.g., `us-central1`).

### How to determine your Location/Region
Your "Location" is the Google Cloud Region where your data will be processed.

**Important Note:** A Google Cloud Project itself doesn't have a single "region". Instead, individual resources and API calls are tied to specific regions. You just need to choose a region where Vertex AI Generative AI is supported.

1. **For new Vertex AI usage:** The most common and feature-rich default is **`us-central1`** (Iowa, USA). It is highly recommended to use this unless you have a specific reason not to.
2. **For existing resources:** If your application (e.g., a Cloud Run service or Compute Engine VM) is already hosted in GCP, it's best to use that same region for Vertex AI to minimize latency (e.g., if your app is in `europe-west4`, use `europe-west4` for Vertex AI).
3. **Checking region availability:** If you require a different region, ensure that the specific Gemini model you are trying to use is supported in that region. You can check the [Vertex AI Generative AI locations documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations) for model availability per region.

**Via Environment Variables (`.env`):**
```env
# Point to your downloaded service account key
GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/simpleai-vertex-key.json"

# Switch Gemini adapter to Gemini Enterprise Agent Platform (GCP / Vertex)
GEMINI_USE_ENTERPRISE=true
GEMINI_ENTERPRISE_PROJECT=your-google-cloud-project-id
GEMINI_ENTERPRISE_LOCATION=us-central1
GEMINI_ENTERPRISE_GCS_BUCKET=your-upload-bucket
GEMINI_ENTERPRISE_GCS_PREFIX=simpleai-uploads
GEMINI_ENTERPRISE_GCS_CLEANUP=always
```

SimpleAI also accepts the Google GenAI SDK environment names for the core switch/project/location:

```env
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=your-google-cloud-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

**Via Django `settings.py` / `ai_settings.json`:**
```json
{
  "providers": {
    "gemini": {
      "use_enterprise": true,
      "enterprise_project": "your-google-cloud-project-id",
      "enterprise_location": "us-central1",
      "enterprise_gcs_bucket": "your-upload-bucket",
      "enterprise_gcs_prefix": "simpleai-uploads",
      "enterprise_gcs_cleanup": "always"
    }
  }
}
```
*(Make sure `GOOGLE_APPLICATION_CREDENTIALS` is still set in the environment where the app is running!)*

## Legacy setting names

Older configs may still use `use_vertexai`, `vertexai_*` keys, or `GEMINI_USE_VERTEXAI` / `GEMINI_VERTEXAI_*` env vars. Those continue to work; prefer the `enterprise_*` names for new projects.

## Enterprise (GCP) file upload behavior

- If `use_enterprise=true` and `enterprise_gcs_bucket` is configured, `simpleai` uploads local files to GCS and passes `gs://...` URIs to Gemini (true binary path).
- If `enterprise_gcs_bucket` is not configured, `simpleai` falls back to extracting file text and appending it to the prompt.
- `enterprise_gcs_cleanup` controls object cleanup:
  - `always` (default): delete uploaded objects after each request, whether success or failure.
  - `on_success`: delete only when model generation succeeds.
  - `never`: keep objects in the bucket.

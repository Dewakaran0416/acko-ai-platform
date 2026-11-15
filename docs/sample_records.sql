-- Sample records of the Acko AI Platform: a few real rows of every table, written by `python -m scripts.export_sample_records`.
-- Taken from a sqlite database on 15 Nov 2026.
-- Rows in the whole database (demo rows not counted): users 3, quotations 1, claims 1, chat_logs 4.
-- People are anonymous: names become Customer 1, Customer 2 ..., e-mail addresses user1@example.com ...; a phone number,
-- a policy number and a vehicle registration keep only their last two characters.
-- Load it after the schema:  psql -f docs/schema_postgresql.sql -f docs/sample_records.sql
BEGIN;

-- users (3 rows: the people of the rows below)
INSERT INTO users (id, name, email, phone, role, created_at) VALUES ('0a44425d-3196-4136-989b-f33bedeed077', 'Customer 1', 'user1@example.com', NULL, 'customer', '2026-11-13 20:00:01.775355');
INSERT INTO users (id, name, email, phone, role, created_at) VALUES ('ff087448-9123-4c91-b08e-abf0504113b1', 'Customer 2', 'user2@example.com', NULL, 'customer', '2026-11-13 22:43:10.742193');
INSERT INTO users (id, name, email, phone, role, created_at) VALUES ('0c46d2ca-742b-4aad-9a4f-4f9c6f5b1848', 'Acko Manager', 'manager@example.com', NULL, 'manager', '2026-11-15 11:48:56.629693');

-- quotations (1 of 1 rows)
INSERT INTO quotations (id, user_id, vehicle_type, vehicle_make, vehicle_model, manufacturing_year, city, idv, ncb_percent, predicted_premium, fuel_type, policy_type, created_at) VALUES ('3d6bf6fc-a45e-4b02-926c-2928aba456f6', 'ff087448-9123-4c91-b08e-abf0504113b1', 'car', 'Honda', 'City', 2020, 'Chennai', 808000.0, 0, 54400.46, 'Petrol', 'Comprehensive', '2026-11-13 22:52:27.705801');

-- claims (1 of 1 rows)
INSERT INTO claims (id, user_id, vehicle_type, policy_number, incident_date, damage_type, affected_parts, damage_severity, image_s3_key, form_s3_key, predicted_amount, approval_probability, status, city, vehicle_make, vehicle_model, idv, vehicle_registration, damage_description, created_at) VALUES ('166cb0ea-c908-4e59-a66f-2a8816cbea17', 'ff087448-9123-4c91-b08e-abf0504113b1', 'car', '*************13', '2026-11-02', 'panel damage', 'Left fender, Front bumper', 'moderate', 'claims/ff087448-9123-4c91-b08e-abf0504113b1/166cb0ea-c908-4e59-a66f-2a8816cbea17/damage.jpg', 'claims/ff087448-9123-4c91-b08e-abf0504113b1/166cb0ea-c908-4e59-a66f-2a8816cbea17/form.pdf', 253127.41, 0.9837, 'approved', 'Chennai', 'Hyundai', 'Creta SX', 850000.0, '********34', 'I hit a road divider while turning at a junction. The front bumper is cracked and the left headlight is broken.', '2026-11-14 11:45:26.228011');

-- chat_logs (4 of 4 rows)
INSERT INTO chat_logs (id, user_id, intent, question, retrieved_source, response, created_at) VALUES ('32a016a1-94e3-4d99-95e3-3e5f3e76e804', '0a44425d-3196-4136-989b-f33bedeed077', 'policy_qa', 'What is the waiting period for Acko health insurance', 'Acko_Health_Insurance_Policy_TC.pdf p.12; Acko_Health_Insurance_Policy_TC.pdf p.17; Acko_Insurance_FAQs.pdf p.10', 'Acko health insurance has five key waiting periods:

*   **Initial/Cooling-off:** 30 days (no illness claims, but accidental injuries are covered from Day 1).
*   **Pre-Existing Disease (PED):** 36 months (standard) or 12 months (Platinum).
*   **Specific Disease:** 24 months for conditions like cataract, hernia, joint replacement, piles, sinusitis, tonsillitis, BPH, spinal disc problems, and gallstones.
*   **Maternity:** 24 months from the date of add-on inclusion.
*   **Bariatric:** Please refer to your specific policy document for details on this waiting period.', '2026-11-13 20:00:20.863618');
INSERT INTO chat_logs (id, user_id, intent, question, retrieved_source, response, created_at) VALUES ('0aeda10b-e485-4c83-b95a-20ebf44c3939', '0a44425d-3196-4136-989b-f33bedeed077', 'claims', 'How do I file a claim for my car', 'Acko_Insurance_FAQs.pdf p.35; Acko_Insurance_FAQs.pdf p.27; Acko_Insurance_FAQs.pdf p.11', 'You can file a motor insurance claim 24x7 using these digital channels:

*   **Acko App (Recommended):** Go to My Policies, select your policy, and tap "File a Claim" to follow the guided steps.
*   **Acko Website:** Log in to your account at acko.com, go to "Claims," and select "File New Claim."
*   **WhatsApp:** Send your policy number and a brief description to Acko''s official WhatsApp business number.
*   **Email:** Send your claim details to claims@acko.com.

After registering, you will receive a Claim Reference Number (CRN) to track your claim.', '2026-11-13 20:00:38.159386');
INSERT INTO chat_logs (id, user_id, intent, question, retrieved_source, response, created_at) VALUES ('8586cb40-d6bb-4e94-bd1f-10ced2ae1933', 'ff087448-9123-4c91-b08e-abf0504113b1', 'quotation', 'How much will insurance cost for my 2020 Honda City in Chennai?', NULL, 'I can give you an instant premium estimate. It only needs your vehicle details (type, make, model, year, city, IDV and No Claim Bonus). Open **Get a Quote** in the menu on the left.', '2026-11-13 22:52:08.371620');
INSERT INTO chat_logs (id, user_id, intent, question, retrieved_source, response, created_at) VALUES ('13b6a59e-4034-4be5-a578-2ca6fdb45315', '0c46d2ca-742b-4aad-9a4f-4f9c6f5b1848', 'policy_qa', 'What add-ons are available for bike insurance?', 'Acko_Insurance_FAQs.pdf p.6; Acko_Insurance_FAQs.pdf p.6; Acko_Insurance_FAQs.pdf p.16', 'For bike insurance, you can opt for the following add-ons:

*   **Pillion Rider PA Add-on:** Covers your pillion passenger for accidental death (INR 1,00,000) and permanent total or partial disability.
*   **Engine Protection:** Covers consequential engine damage, such as water ingression or oil starvation, which is excluded from standard policies.
*   **Roadside Assistance (RSA):** Provides help during breakdowns.
*   **Zero Depreciation:** Prevents a 30–50% deduction on parts during claims.
*   **Key Protection:** Covers costs related to modern smart keys.

Some of these can be added mid-year as an endorsement by paying a pro-rata premium. Please check the Acko app for specific availability.', '2026-11-15 11:50:05.912587');

COMMIT;

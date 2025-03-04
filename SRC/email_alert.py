import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
from dotenv import load_dotenv

load_dotenv()

# Send email with customer churn table for the selected state

sender_email = os.getenv('SENDER_EMAIL')
auth_password = os.getenv('AUTH_PASSWORD')
smtp_server = os.getenv('SMTP_SERVER')

def send_email(selected_state, filtered_results, rec_email):
    if not rec_email:
        st.error("Please enter a valid email address.")
        return

    msg = MIMEMultipart('alternative')
    msg['From'] = sender_email
    msg['To'] = rec_email
    msg['Subject'] = f"Your Churn Alert for {selected_state}"

    top_churn_customers = filtered_results.nlargest(10, 'Churn Probability')[['Churn Probability', 'Churn Prediction']]
    
    html_table = top_churn_customers.to_html(index=False, justify='center', float_format='{:,.2%}'.format)

    # Create plain and HTML versions of the email
    plain_text = f"""
    Dear user,

    Please find attached the current top 10 customers most likely to churn in {selected_state}.

    Best regards,
    Churn Prediction Team
    """

    html_content = f"""
    <html>
      <body>
        <p>Dear user,</p>
        <p>Please find below the current top 10 customers most likely to churn in {selected_state}:</p>
        {html_table}
        <p>Best regards,<br>Churn Prediction Team</p>
      </body>
    </html>
    """

    msg.attach(MIMEText(plain_text, 'plain'))
    msg.attach(MIMEText(html_content, 'html'))

    try:
      server = smtplib.SMTP(smtp_server, 587)
      server.ehlo()
      server.starttls()
      server.ehlo()
      server.login(sender_email, auth_password)
      server.sendmail(sender_email, rec_email, msg.as_string())
      server.quit()
      return True
    except Exception as e:
      return e
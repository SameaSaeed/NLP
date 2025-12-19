# Create the dataset file
cat > create_dataset.py << 'EOF'
import pandas as pd
import random

# Sample EdTech feedback data
positive_feedback = [
    "This learning platform is amazing! My students love the interactive features.",
    "Great user interface and easy navigation. Highly recommend for educators.",
    "The video lessons are clear and engaging. Students show improved performance.",
    "Excellent customer support and regular updates. Very satisfied with the service.",
    "Interactive quizzes make learning fun and effective for my classroom.",
    "The analytics dashboard helps me track student progress efficiently.",
    "Love the collaborative features that allow students to work together.",
    "The mobile app works perfectly and students can learn anywhere.",
    "Content is well-structured and aligns with curriculum standards.",
    "The gamification elements keep students motivated and engaged.",
    "Easy to integrate with our existing school management system.",
    "The assessment tools are comprehensive and save me lots of time.",
    "Students enjoy the personalized learning paths and adaptive content.",
    "The platform is reliable and rarely experiences downtime.",
    "Great value for money compared to other educational platforms.",
    "The reporting features help me communicate progress to parents effectively.",
    "Students have shown significant improvement since using this platform.",
    "The content library is extensive and covers all required topics.",
    "User-friendly design makes it easy for both teachers and students.",
    "The platform adapts well to different learning styles and needs."
]

negative_feedback = [
    "The platform is too slow and crashes frequently during lessons.",
    "Very confusing interface. Students struggle to navigate the system.",
    "Poor customer service response time. Issues take weeks to resolve.",
    "The content is outdated and doesn't match current curriculum standards.",
    "Too expensive for the limited features provided. Not worth the cost.",
    "The mobile app is buggy and doesn't sync properly with the web version.",
    "Lack of interactive elements makes the lessons boring for students.",
    "The assessment tools are limited and don't provide detailed feedback.",
    "Difficult to integrate with our school's existing technology infrastructure.",
    "The platform lacks essential features that competitors offer.",
    "Students complain about the complicated login process and frequent logouts.",
    "The video quality is poor and audio often cuts out during presentations.",
    "Limited customization options for different grade levels and subjects.",
    "The analytics are basic and don't provide actionable insights.",
    "Frequent technical issues disrupt classroom activities and learning.",
    "The content is not engaging enough to hold student attention.",
    "Poor accessibility features for students with special needs.",
    "The platform doesn't work well with older devices in our school.",
    "Lack of offline functionality makes it unusable in areas with poor internet.",
    "The grading system is inflexible and doesn't match our school's requirements."
]

# Create balanced dataset
data = []
labels = []

# Add positive feedback
for feedback in positive_feedback:
    data.append(feedback)
    labels.append(1)  # 1 for positive

# Add negative feedback
for feedback in negative_feedback:
    data.append(feedback)
    labels.append(0)  # 0 for negative

# Create DataFrame
df = pd.DataFrame({
    'feedback': data,
    'sentiment': labels
})

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Save to CSV
df.to_csv('data/edtech_feedback.csv', index=False)
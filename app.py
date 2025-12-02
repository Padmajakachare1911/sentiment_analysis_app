"""
Main Streamlit Application for Sentiment Analysis with Active Learning
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
from database import Database
from ml_models import SentimentAnalyzer, AspectExtractor, TopicModeler, process_reviews
from active_learning import ActiveLearning

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'current_project' not in st.session_state:
    st.session_state.current_project = None

# Initialize database and active learning
db = Database()
active_learning = ActiveLearning(db)


def login_page():
    """Login/Signup page"""
    st.title("🔐 Login / Signup")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            user = db.authenticate_user(username, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.user = user
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with tab2:
        st.subheader("Create Account")
        new_username = st.text_input("Username", key="signup_username")
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_password")
        role = st.selectbox("Role", ["user", "admin"], key="signup_role")
        
        if st.button("Sign Up"):
            if new_username and new_email and new_password:
                user_id = db.create_user(new_username, new_email, new_password, role)
                if user_id:
                    st.success("Account created successfully! Please login.")
                else:
                    st.error("Username or email already exists")
            else:
                st.error("Please fill all fields")


def upload_dataset_page():
    """Upload dataset page"""
    st.title("📤 Upload Dataset")
    
    # Project selection/creation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Select Project")
        projects = db.get_user_projects(st.session_state.user['id'])
        if projects:
            project_names = [f"{p['name']} (ID: {p['id']})" for p in projects]
            selected_project = st.selectbox("Choose a project", project_names)
            if selected_project:
                project_id = int(selected_project.split("ID: ")[1].split(")")[0])
                st.session_state.current_project = project_id
        else:
            st.info("No projects found. Create a new project below.")
    
    with col2:
        st.subheader("Create New Project")
        new_project_name = st.text_input("Project Name")
        new_project_desc = st.text_area("Description")
        if st.button("Create Project"):
            if new_project_name:
                project_id = db.create_project(new_project_name, new_project_desc, st.session_state.user['id'])
                st.success(f"Project '{new_project_name}' created!")
                st.rerun()
    
    st.divider()
    
    # Dataset upload
    st.subheader("Upload Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], help="CSV must contain a 'review' column")
    
    with col2:
        if st.button("📊 Load Sample Dataset", use_container_width=True):
            # Generate sample dataset
            sample_data = {
                'review': [
                    "This product is amazing! I love it so much.",
                    "Terrible quality, very disappointed.",
                    "It's okay, nothing special.",
                    "Best purchase I've ever made!",
                    "Waste of money, don't buy this.",
                    "The service was good but could be better.",
                    "Absolutely fantastic experience!",
                    "Poor customer service, very rude staff.",
                    "Decent product for the price.",
                    "Outstanding quality and fast delivery!"
                ]
            }
            df = pd.DataFrame(sample_data)
            st.session_state.sample_df = df
            st.success("Sample dataset loaded!")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'review' not in df.columns:
                st.error("CSV file must contain a 'review' column")
            else:
                st.session_state.uploaded_df = df
                st.success(f"Dataset loaded: {len(df)} reviews")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Process dataset
    if 'uploaded_df' in st.session_state or 'sample_df' in st.session_state:
        df = st.session_state.get('uploaded_df') or st.session_state.get('sample_df')
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        if st.session_state.current_project:
            if st.button("🚀 Process Dataset", type="primary", use_container_width=True):
                with st.spinner("Processing reviews..."):
                    reviews = df['review'].dropna().tolist()
                    
                    # Initialize models
                    analyzer = SentimentAnalyzer(method='vader')
                    aspect_extractor = AspectExtractor()
                    
                    # Process reviews
                    results = process_reviews(reviews, analyzer, aspect_extractor)
                    
                    # Save to database
                    db.insert_sentiment_results(st.session_state.current_project, results)
                    
                    st.success(f"Processed {len(results)} reviews!")
                    st.balloons()
        else:
            st.warning("Please select or create a project first")


def dashboard_page():
    """Dashboard page with visualizations"""
    st.title("📊 Dashboard")
    
    # Project selection
    projects = db.get_user_projects(st.session_state.user['id'])
    if not projects:
        st.warning("No projects found. Please upload a dataset first.")
        return
    
    project_options = ["All Projects"] + [f"{p['name']} (ID: {p['id']})" for p in projects]
    selected_project = st.selectbox("Select Project", project_options)
    
    project_id = None
    if selected_project != "All Projects":
        project_id = int(selected_project.split("ID: ")[1].split(")")[0])
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_filter = st.date_input("Date Range", value=[datetime.now() - timedelta(days=30), datetime.now()])
    
    with col2:
        sentiment_filter = st.multiselect("Sentiment", ["positive", "negative", "neutral"], default=["positive", "negative", "neutral"])
    
    with col3:
        min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.0)
    
    st.divider()
    
    # Statistics
    stats = db.get_sentiment_stats(project_id)
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_reviews = sum(s['count'] for s in stats.values())
    
    with col1:
        st.metric("Total Reviews", total_reviews)
    
    with col2:
        positive_count = stats.get('positive', {}).get('count', 0)
        st.metric("Positive", positive_count, delta=f"{positive_count/total_reviews*100:.1f}%" if total_reviews > 0 else "0%")
    
    with col3:
        negative_count = stats.get('negative', {}).get('count', 0)
        st.metric("Negative", negative_count, delta=f"{negative_count/total_reviews*100:.1f}%" if total_reviews > 0 else "0%")
    
    with col4:
        neutral_count = stats.get('neutral', {}).get('count', 0)
        st.metric("Neutral", neutral_count, delta=f"{neutral_count/total_reviews*100:.1f}%" if total_reviews > 0 else "0%")
    
    st.divider()
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Distribution")
        if stats:
            sentiment_df = pd.DataFrame([
                {'Sentiment': k, 'Count': v['count']}
                for k, v in stats.items()
            ])
            fig = px.bar(sentiment_df, x='Sentiment', y='Count', color='Sentiment',
                        color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'gray'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available")
    
    with col2:
        st.subheader("Confidence Distribution")
        # Get confidence data
        conn = db.get_connection()
        if project_id:
            query = "SELECT confidence FROM sentiment_results WHERE project_id = ?"
            conf_df = pd.read_sql_query(query, conn, params=(project_id,))
        else:
            query = "SELECT confidence FROM sentiment_results"
            conf_df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not conf_df.empty:
            fig = px.histogram(conf_df, x='confidence', nbins=20, title="Confidence Score Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available")
    
    # Sentiment over time
    st.subheader("Sentiment Trends Over Time")
    time_df = db.get_sentiment_over_time(project_id)
    
    if not time_df.empty:
        time_df['date'] = pd.to_datetime(time_df['date'])
        fig = px.line(time_df, x='date', y='count', color='sentiment',
                     title="Sentiment Over Time",
                     color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'gray'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No time-series data available")
    
    # Aspect sentiment
    st.subheader("Aspect Sentiment Analysis")
    conn = db.get_connection()
    if project_id:
        query = """
            SELECT aspect, sentiment, COUNT(*) as count
            FROM sentiment_results
            WHERE project_id = ? AND aspect IS NOT NULL
            GROUP BY aspect, sentiment
            ORDER BY count DESC
            LIMIT 20
        """
        aspect_df = pd.read_sql_query(query, conn, params=(project_id,))
    else:
        query = """
            SELECT aspect, sentiment, COUNT(*) as count
            FROM sentiment_results
            WHERE aspect IS NOT NULL
            GROUP BY aspect, sentiment
            ORDER BY count DESC
            LIMIT 20
        """
        aspect_df = pd.read_sql_query(query, conn)
    conn.close()
    
    if not aspect_df.empty:
        fig = px.bar(aspect_df, x='aspect', y='count', color='sentiment',
                    color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'gray'},
                    title="Top Aspects by Sentiment")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No aspect data available")


def active_learning_page():
    """Active Learning page for manual annotation"""
    st.title("🎯 Active Learning")
    st.markdown("### Review and Correct Low Confidence Predictions")
    
    # Project selection
    projects = db.get_user_projects(st.session_state.user['id'])
    if not projects:
        st.warning("No projects found. Please upload a dataset first.")
        return
    
    project_options = ["All Projects"] + [f"{p['name']} (ID: {p['id']})" for p in projects]
    selected_project = st.selectbox("Select Project", project_options, key="al_project")
    
    project_id = None
    if selected_project != "All Projects":
        project_id = int(selected_project.split("ID: ")[1].split(")")[0])
    
    # Get uncertain samples
    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.50, 0.05, key="al_threshold")
    
    uncertain_samples = active_learning.get_uncertain_samples(project_id, threshold)
    
    st.metric("Uncertain Samples", len(uncertain_samples))
    
    if len(uncertain_samples) == 0:
        st.success("🎉 No uncertain samples found! All predictions have high confidence.")
        return
    
    st.divider()
    
    # Display samples in a styled table format
    st.subheader("Low Confidence Reviews")
    
    # Initialize session state for corrections
    if 'corrections' not in st.session_state:
        st.session_state.corrections = {}
    
    # Display each sample
    for idx, sample in enumerate(uncertain_samples[:50]):  # Limit to 50 for performance
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                # Confidence badge
                confidence_color = "🔴" if sample['confidence'] < 0.3 else "🟡" if sample['confidence'] < 0.4 else "🟠"
                st.markdown(f"**{confidence_color} Confidence: {sample['confidence']:.2%}** | "
                          f"**Predicted: {sample['sentiment'].upper()}**")
                
                # Review text
                st.text_area(
                    "Review Text",
                    value=sample['review_text'],
                    height=80,
                    key=f"review_{sample['id']}",
                    disabled=True
                )
                
                if sample.get('aspect'):
                    st.caption(f"📍 Aspect: {sample['aspect']}")
            
            with col2:
                st.markdown("### Correct Sentiment")
                corrected_sentiment = st.selectbox(
                    "Sentiment",
                    ["positive", "negative", "neutral"],
                    index=["positive", "negative", "neutral"].index(sample['sentiment']),
                    key=f"sentiment_{sample['id']}"
                )
                
                if st.button("💾 Save Correction & Add to Training Set", 
                           key=f"save_{sample['id']}", 
                           use_container_width=True,
                           type="primary"):
                    active_learning.save_correction(
                        sample['id'],
                        corrected_sentiment,
                        st.session_state.user['id']
                    )
                    st.success("✅ Correction saved and added to training set!")
                    st.rerun()
            
            with col3:
                st.markdown("### Info")
                st.caption(f"**ID:** {sample['id']}")
                if sample.get('project_name'):
                    st.caption(f"**Project:** {sample['project_name']}")
                st.caption(f"**Date:** {sample['created_at']}")
        
        st.divider()
    
    # Batch retrain button
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🚀 Re-annotate & Retrain Model", type="primary", use_container_width=True):
            with st.spinner("Retraining model with corrections..."):
                result = active_learning.retrain_model(project_id)
                
                if result.get('success'):
                    st.success(f"✅ Model retrained successfully!")
                    st.metric("Accuracy", f"{result['accuracy']:.2%}")
                    st.metric("Training Samples", result['training_samples'])
                    st.metric("Version", result['version'])
                    st.balloons()
                else:
                    st.error(f"Error: {result.get('error', 'Unknown error')}")


def admin_panel_page():
    """Admin panel page"""
    if st.session_state.user.get('role') != 'admin':
        st.error("Access denied. Admin privileges required.")
        return
    
    st.title("⚙️ Admin Panel")
    
    # Get admin stats
    stats = db.get_admin_stats()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Users", stats['active_users'])
    
    with col2:
        st.metric("Total Annotations", stats['total_annotations'])
    
    with col3:
        st.metric("Pending Samples", stats['pending_samples'])
    
    with col4:
        accuracy = stats['model_accuracy']
        if accuracy:
            st.metric("Model Accuracy", f"{accuracy:.2%}")
        else:
            st.metric("Model Accuracy", "N/A")
    
    st.divider()
    
    # Model versions
    st.subheader("Model Versions")
    versions = db.get_model_versions()
    
    if versions:
        versions_df = pd.DataFrame(versions)
        versions_df['created_at'] = pd.to_datetime(versions_df['created_at'])
        versions_df['is_active'] = versions_df['is_active'].apply(lambda x: "✅ Active" if x == 1 else "❌ Inactive")
        
        st.dataframe(
            versions_df[['version_number', 'accuracy', 'training_samples', 'created_at', 'is_active']],
            use_container_width=True
        )
        
        # Model accuracy chart
        if len(versions) > 1:
            fig = px.line(
                versions_df.sort_values('created_at'),
                x='created_at',
                y='accuracy',
                markers=True,
                title="Model Accuracy Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No model versions found")
    
    st.divider()
    
    # Deployment readiness
    st.subheader("Deployment Readiness")
    
    readiness_score = 0
    checks = []
    
    # Check 1: Model exists
    if stats['model_accuracy'] is not None:
        readiness_score += 25
        checks.append(("✅", "Model Trained", "Model exists with accuracy"))
    else:
        checks.append(("❌", "Model Trained", "No trained model found"))
    
    # Check 2: Sufficient annotations
    if stats['total_annotations'] >= 10:
        readiness_score += 25
        checks.append(("✅", "Sufficient Annotations", f"{stats['total_annotations']} annotations"))
    else:
        checks.append(("⚠️", "Sufficient Annotations", f"Only {stats['total_annotations']} annotations"))
    
    # Check 3: Low pending samples
    if stats['pending_samples'] < 50:
        readiness_score += 25
        checks.append(("✅", "Low Pending Samples", f"{stats['pending_samples']} uncertain samples"))
    else:
        checks.append(("⚠️", "Low Pending Samples", f"{stats['pending_samples']} uncertain samples"))
    
    # Check 4: Model accuracy threshold
    if stats['model_accuracy'] and stats['model_accuracy'] >= 0.70:
        readiness_score += 25
        checks.append(("✅", "High Accuracy", f"{stats['model_accuracy']:.2%} accuracy"))
    elif stats['model_accuracy']:
        checks.append(("⚠️", "High Accuracy", f"{stats['model_accuracy']:.2%} accuracy (target: 70%+)"))
    else:
        checks.append(("❌", "High Accuracy", "No accuracy data"))
    
    # Display checks
    for icon, title, desc in checks:
        st.markdown(f"{icon} **{title}**: {desc}")
    
    # Overall readiness
    st.divider()
    st.metric("Deployment Readiness", f"{readiness_score}%")
    
    if readiness_score >= 75:
        st.success("✅ System is ready for deployment!")
    elif readiness_score >= 50:
        st.warning("⚠️ System needs improvement before deployment")
    else:
        st.error("❌ System is not ready for deployment")


# Main app logic
def main():
    """Main application"""
    # Sidebar navigation
    with st.sidebar:
        st.title("📊 Sentiment Analysis")
        
        if st.session_state.authenticated:
            st.success(f"Logged in as: {st.session_state.user['username']}")
            st.caption(f"Role: {st.session_state.user['role']}")
            
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.user = None
                st.session_state.current_project = None
                st.rerun()
            
            st.divider()
            
            # Navigation
            page = st.radio(
                "Navigation",
                ["📤 Upload Dataset", "📊 Dashboard", "🎯 Active Learning", "⚙️ Admin Panel"],
                label_visibility="collapsed"
            )
        else:
            page = "Login"
    
    # Route to appropriate page
    if not st.session_state.authenticated:
        login_page()
    else:
        if page == "📤 Upload Dataset":
            upload_dataset_page()
        elif page == "📊 Dashboard":
            dashboard_page()
        elif page == "🎯 Active Learning":
            active_learning_page()
        elif page == "⚙️ Admin Panel":
            admin_panel_page()


if __name__ == "__main__":
    main()


from django.urls import path
from .views import (
    ListPetLocationsView, ListLostPetsView, ListFoundPetsView,
    ReportPetLocationView, MarkPetStatusView, UserPetLocationsView,
    RegisterView, LoginView, LogoutView, GoogleLoginView, ProfileView, AddPetView, PublicPetDashboardView, EditedPetImageViewSet,
    DeletePetView, SearchPetView, EditPetView, GetPetCountView, GetUserCountView, contact_pet_owner,
    toggle_share_contact_info, share_contact,
    VerifyEmailView, ResendVerificationView, ForgotPasswordView, ResetPasswordView,
    FeatureStatusView, RetryFeaturesView,
    NotificationListView, MarkNotificationReadView, MarkAllNotificationsReadView,
    forward_reply_webhook,
)
from .chat_views import (
    ConversationListView, ConversationDetailView,
    MessageListCreateView, ConversationStartView,
)
from .tag_views import PetQRTagView, PublicPetByAnimalIdView

edited_pet_image_list = EditedPetImageViewSet.as_view({
    'get': 'list',
    'post': 'create'
})

edited_pet_image_detail = EditedPetImageViewSet.as_view({
    'get': 'retrieve',
    'put': 'update',
    'patch': 'partial_update',
    'delete': 'destroy'
})

urlpatterns = [
    # Your existing URLs

    path('signup/', RegisterView.as_view(), name='signup'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('google/', GoogleLoginView.as_view(), name='google_login'),
    path('verify-email/', VerifyEmailView.as_view(), name='verify_email'),
    path('resend-verification/', ResendVerificationView.as_view(), name='resend_verification'),
    path('forgot-password/', ForgotPasswordView.as_view(), name='forgot_password'),
    path('reset-password/', ResetPasswordView.as_view(), name='reset_password'),
    path('profile/', ProfileView.as_view(), name='profile'),
    path('pets/add/', AddPetView.as_view(), name='add_pet'), 
    path('pets/search/', SearchPetView.as_view(), name='search_pet'), 
    path('dashboard/pets/', PublicPetDashboardView.as_view(), name='public_pet_dashboard'),
    path('pets/<int:pet_id>/delete/', DeletePetView.as_view(), name='delete_pet'),
    path('pets/<int:pet_id>/edit/', EditPetView.as_view(), name='edit_pet'),
    path('pets/<int:pet_id>/feature-status/', FeatureStatusView.as_view(), name='feature_status'),
    path('pets/<int:pet_id>/retry-features/', RetryFeaturesView.as_view(), name='retry_features'),
    path('pets/count/', GetPetCountView.as_view(), name='get_pet_count'),  
    path('users/count/', GetUserCountView.as_view(), name='get_user_count'),
    
    # New pet location URLs
    path('pets/locations/', ListPetLocationsView.as_view(), name='pet_locations'),
    path('pets/lost/locations/', ListLostPetsView.as_view(), name='lost_pets'),
    path('pets/found/locations/', ListFoundPetsView.as_view(), name='found_pets'),
    path('pets/report/', ReportPetLocationView.as_view(), name='report_pet'),
    path('pets/locations/<int:location_id>/status/', MarkPetStatusView.as_view(), name='update_pet_status'),
    path('user/pet-locations/', UserPetLocationsView.as_view(), name='user_pet_locations'),
    path('pets/contact-owner/', contact_pet_owner, name='contact_pet_owner'),
    path('email/forward-reply/', forward_reply_webhook, name='forward_reply_webhook'),
    path('api/auth/conversations/share-info/', toggle_share_contact_info, name='toggle_share_contact_info'),
    path('share-contact/<uuid:conversation_id>/<str:user_type>/<str:decision>/', share_contact, name='share_contact'),   
    path('notifications/', NotificationListView.as_view(), name='notifications'),
    path('notifications/<int:notification_id>/read/', MarkNotificationReadView.as_view(), name='mark_notification_read'),
    path('notifications/read-all/', MarkAllNotificationsReadView.as_view(), name='mark_all_notifications_read'),
    path('edited-pet-images/', edited_pet_image_list, name='edited_pet_image_list'),
    path('edited-pet-images/<int:pk>/', edited_pet_image_detail, name='edited_pet_image_detail'),

    # In-app chat
    path('chat/conversations/', ConversationListView.as_view(), name='chat_conversations'),
    path('chat/conversations/start/', ConversationStartView.as_view(), name='chat_conversations_start'),
    path('chat/conversations/<uuid:conversation_id>/', ConversationDetailView.as_view(), name='chat_conversation_detail'),
    path('chat/conversations/<uuid:conversation_id>/messages/', MessageListCreateView.as_view(), name='chat_messages'),

    # QR pet tags
    path('pets/<int:pet_id>/qr/', PetQRTagView.as_view(), name='pet_qr_tag'),
    # Public landing page for scanned QR codes (no auth)
    path('found/<str:animal_id>/', PublicPetByAnimalIdView.as_view(), name='public_pet_by_animal_id'),
]

jQuery( document ).ready(function( $ ) {
    // Togglable side navigation.
    $( "#menuLink" ).click(function( event ) {
        event.preventDefault();
        $( "#layout" ).toggleClass( "active" );
        $( "#menu" ).toggleClass( "active" );
        $( "#menuLink" ).toggleClass( "active" );
    });

    // Enable linking for buttons with a `data-href` attribute.
    $('button[data-href]').click(function(event) {
        window.location.href = $(this).data("href");
    });
});
